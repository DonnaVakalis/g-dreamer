"""
Dreamer agent for the encoder-swap graph ablation.

This module intentionally mirrors upstream DreamerV3 very closely.
The important design choice is not "we rewrote Dreamer", but the opposite:
we kept Dreamer's RSSM, decoder, losses, actor, critic, and training loop
effectively unchanged so that the only substantive experimental difference is
the observation encoder.

Why the custom shell exists at all:
  The graph-encoder variant needs structured graph keys for encoding, but we
  still want the reconstruction loss to stay anchored to the baseline's flat
  `vector` observation. Upstream Dreamer uses the same observation dict for
  both encoder and decoder, so we introduce a thin wrapper that makes those
  key choices explicit:

    baseline-like encoder swap:
      encoder sees: vector
      decoder reconstructs: vector

    graph encoder swap:
      encoder sees: nodes/senders/receivers/node_mask/edge_mask
      decoder reconstructs: vector

That keeps the loss surface and logging comparable while isolating the question
"does graph structure help the representation before RSSM surgery?"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from dgr.agents.graph_dreamerv3.encoders.gnn import GraphEncoder


def _ensure_upstream_on_path() -> None:
    upstream_root = Path(__file__).resolve().parents[4] / "third_party" / "dreamerv3"
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))


_ensure_upstream_on_path()
from dreamerv3 import rssm  # noqa: E402
from dreamerv3.agent import imag_loss, repl_loss  # noqa: E402

f32 = jnp.float32
i32 = jnp.int32


def sg(xs, skip=False):
    return xs if skip else jax.lax.stop_gradient(xs)


def sample(xs):
    return jax.tree.map(lambda x: x.sample(nj.seed()), xs)


def prefix(xs, p):
    return {f"{p}/{k}": v for k, v in xs.items()}


def concat(xs, a):
    return jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)


def isimage(s):
    return s.dtype == np.uint8 and len(s.shape) == 3


_CONTROL_KEYS = ("is_first", "is_last", "is_terminal", "reward")
_GRAPH_KEYS = ("nodes", "senders", "receivers", "node_mask", "edge_mask")
_VECTOR_KEYS = ("vector",)


class Agent(embodied.jax.Agent):
    banner = [
        r"---   ____                 __      ___   __ ---",
        r"---  / ___|_ __ __ _ _ __ \ \    / / | / / ---",
        r"--- | |  _| '__/ _` | '_ \ \ \/\/ /| |/ /  ---",
        r"--- | |_| | | | (_| | |_) | \_/\_/ |___/   ---",
        r"---  \____|_|  \__,_| .__/                    ---",
        r"---                 |_|                       ---",
    ]

    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config

        enc_space = self._encoder_obs_space(obs_space, config.enc.typ)
        dec_space = self._decoder_obs_space(obs_space)

        enc_ctor, enc_kw = self._make_encoder_ctor(config)
        self.enc = enc_ctor(enc_space, **enc_kw, name="enc")
        self.dyn = {
            "rssm": rssm.RSSM,
        }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name="dyn")
        self.dec = {
            "simple": rssm.Decoder,
        }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name="dec")

        self.feat2tensor = lambda x: jnp.concatenate(
            [nn.cast(x["deter"]), nn.cast(x["stoch"].reshape((*x["stoch"].shape[:-2], -1)))], -1
        )

        scalar = elements.Space(np.float32, ())
        binary = elements.Space(bool, (), 0, 2)
        self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name="rew")
        self.con = embodied.jax.MLPHead(binary, **config.conhead, name="con")

        d1, d2 = config.policy_dist_disc, config.policy_dist_cont
        outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
        self.pol = embodied.jax.MLPHead(act_space, outs, **config.policy, name="pol")

        self.val = embodied.jax.MLPHead(scalar, **config.value, name="val")
        self.slowval = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name="slowval"),
            source=self.val,
            **config.slowvalue,
        )

        self.retnorm = embodied.jax.Normalize(**config.retnorm, name="retnorm")
        self.valnorm = embodied.jax.Normalize(**config.valnorm, name="valnorm")
        self.advnorm = embodied.jax.Normalize(**config.advnorm, name="advnorm")

        self.modules = [self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
        self.opt = embodied.jax.Optimizer(
            self.modules,
            self._make_opt(**config.opt),
            summary_depth=1,
            name="opt",
        )

        scales = self.config.loss_scales.copy()
        rec = scales.pop("rec")
        scales.update({k: rec for k in dec_space})
        self.scales = scales

    @staticmethod
    def _encoder_obs_space(obs_space, enc_typ: str):
        if enc_typ == "simple":
            return Agent._pick_obs_space(obs_space, _VECTOR_KEYS)
        if enc_typ == "graph":
            return Agent._pick_obs_space(obs_space, _GRAPH_KEYS)
        raise NotImplementedError(f"Unknown encoder type: {enc_typ}")

    @staticmethod
    def _decoder_obs_space(obs_space):
        # The decoder stays anchored to the flat vector reconstruction target so
        # that encoder ablations do not silently change the training objective.
        return Agent._pick_obs_space(obs_space, _VECTOR_KEYS)

    @staticmethod
    def _pick_obs_space(obs_space, keys):
        picked = {k: obs_space[k] for k in keys if k in obs_space}
        if set(picked) != set(keys):
            missing = sorted(set(keys) - set(picked))
            raise ValueError(f"Missing required observation keys: {missing}")
        return picked

    def _make_encoder_ctor(self, config):
        if config.enc.typ == "simple":
            return rssm.Encoder, config.enc.simple
        if config.enc.typ == "graph":
            return GraphEncoder, config.enc.simple
        raise NotImplementedError(config.enc.typ)

    @property
    def policy_keys(self):
        return "^(enc|dyn|dec|pol)/"

    @property
    def ext_space(self):
        spaces = {}
        spaces["consec"] = elements.Space(np.int32)
        spaces["stepid"] = elements.Space(np.uint8, 20)
        if self.config.replay_context:
            spaces.update(
                elements.tree.flatdict(
                    dict(
                        enc=self.enc.entry_space,
                        dyn=self.dyn.entry_space,
                        dec=self.dec.entry_space,
                    )
                )
            )
        return spaces

    def init_policy(self, batch_size):
        def zeros(x):
            return jnp.zeros((batch_size, *x.shape), x.dtype)

        return (
            self.enc.initial(batch_size),
            self.dyn.initial(batch_size),
            self.dec.initial(batch_size),
            jax.tree.map(zeros, self.act_space),
        )

    def init_train(self, batch_size):
        return self.init_policy(batch_size)

    def init_report(self, batch_size):
        return self.init_policy(batch_size)

    def policy(self, carry, obs, mode="train"):
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        kw = dict(training=False, single=True)
        reset = obs["is_first"]
        enc_obs = self._select_encoder_obs(obs)
        enc_carry, enc_entry, tokens = self.enc(enc_carry, enc_obs, reset, **kw)
        dyn_carry, dyn_entry, feat = self.dyn.observe(dyn_carry, tokens, prevact, reset, **kw)
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, recons = self.dec(
                dec_carry,
                feat,
                reset,
                **kw,
            )
            del recons
        policy = self.pol(self.feat2tensor(feat), bdims=1)
        act = sample(policy)
        out = {}
        out["finite"] = elements.tree.flatdict(
            jax.tree.map(
                lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
                dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act),
            )
        )
        carry = (enc_carry, dyn_carry, dec_carry, act)
        if self.config.replay_context:
            out.update(elements.tree.flatdict(dict(enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
        return carry, act, out

    def train(self, carry, data):
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        metrics, (carry, entries, outs, mets) = self.opt(
            self.loss,
            carry,
            obs,
            prevact,
            training=True,
            has_aux=True,
        )
        metrics.update(mets)
        self.slowval.update()
        outs = {}
        if self.config.replay_context:
            updates = elements.tree.flatdict(
                dict(
                    stepid=stepid,
                    enc=entries[0],
                    dyn=entries[1],
                    dec=entries[2],
                )
            )
            bsize, tlen = obs["is_first"].shape
            assert all(x.shape[:2] == (bsize, tlen) for x in updates.values()), (
                (bsize, tlen),
                {k: v.shape for k, v in updates.items()},
            )
            outs["replay"] = updates
        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, outs, metrics

    def loss(self, carry, obs, prevact, training):
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs["is_first"]
        bsize, tlen = reset.shape
        losses = {}
        metrics = {}

        enc_obs = self._select_encoder_obs(obs)
        dec_obs = self._select_decoder_obs(obs)

        enc_carry, enc_entries, tokens = self.enc(enc_carry, enc_obs, reset, training)
        dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
            dyn_carry,
            tokens,
            prevact,
            reset,
            training,
        )
        losses.update(los)
        metrics.update(mets)
        dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)
        inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
        losses["rew"] = self.rew(inp, 2).loss(obs["reward"])
        con = f32(~obs["is_terminal"])
        if self.config.contdisc:
            con *= 1 - 1 / self.config.horizon
        losses["con"] = self.con(self.feat2tensor(repfeat), 2).loss(con)
        for key, recon in recons.items():
            space, value = self.obs_space[key], dec_obs[key]
            assert value.dtype == space.dtype, (key, space, value.dtype)
            target = f32(value) / 255 if isimage(space) else value
            losses[key] = recon.loss(sg(target))

        shapes = {k: v.shape for k, v in losses.items()}
        assert all(x == (bsize, tlen) for x in shapes.values()), ((bsize, tlen), shapes)

        k_last = min(self.config.imag_last or tlen, tlen)
        horizon = self.config.imag_length
        starts = self.dyn.starts(dyn_entries, dyn_carry, k_last)

        def policyfn(feat):
            return sample(self.pol(self.feat2tensor(feat), 1))

        _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, horizon, training)
        first = jax.tree.map(
            lambda x: x[:, -k_last:].reshape((bsize * k_last, 1, *x.shape[2:])),
            repfeat,
        )
        imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
        lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
        lastact = jax.tree.map(lambda x: x[:, None], lastact)
        imgact = concat([imgprevact, lastact], 1)
        assert all(x.shape[:2] == (bsize * k_last, horizon + 1) for x in jax.tree.leaves(imgfeat))
        assert all(x.shape[:2] == (bsize * k_last, horizon + 1) for x in jax.tree.leaves(imgact))
        inp = self.feat2tensor(imgfeat)
        los, imgloss_out, mets = imag_loss(
            imgact,
            self.rew(inp, 2).pred(),
            self.con(inp, 2).prob(1),
            self.pol(inp, 2),
            self.val(inp, 2),
            self.slowval(inp, 2),
            self.retnorm,
            self.valnorm,
            self.advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
        )
        losses.update({k: v.mean(1).reshape((bsize, k_last)) for k, v in los.items()})
        metrics.update(mets)

        if self.config.repval_loss:
            feat = sg(repfeat, skip=self.config.repval_grad)
            last, term, rew = [obs[k] for k in ("is_last", "is_terminal", "reward")]
            boot = imgloss_out["ret"][:, 0].reshape(bsize, k_last)
            feat, last, term, rew, boot = jax.tree.map(
                lambda x: x[:, -k_last:],
                (feat, last, term, rew, boot),
            )
            inp = self.feat2tensor(feat)
            los, reploss_out, mets = repl_loss(
                last,
                term,
                rew,
                boot,
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.valnorm,
                update=training,
                horizon=self.config.horizon,
                **self.config.repl_loss,
            )
            losses.update(los)
            metrics.update(prefix(mets, "reploss"))
            del reploss_out

        assert set(losses.keys()) == set(self.scales.keys()), (
            sorted(losses.keys()),
            sorted(self.scales.keys()),
        )
        metrics.update({f"loss/{k}": v.mean() for k, v in losses.items()})
        loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {"tokens": tokens, "repfeat": repfeat, "losses": losses}
        return loss, (carry, entries, outs, metrics)

    def report(self, carry, data):
        if not self.config.report:
            return carry, {}

        carry, obs, prevact, _ = self._apply_replay_context(carry, data)
        (enc_carry, dyn_carry, dec_carry) = carry
        bsize, tlen = obs["is_first"].shape
        report_batch = min(6, bsize)
        metrics = {}

        _, (new_carry, entries, outs, mets) = self.loss(carry, obs, prevact, training=False)
        metrics.update(mets)

        if self.config.report_gradnorms:
            for key in self.scales:
                try:

                    def lossfn(data, carry):
                        del data
                        return self.loss(
                            carry,
                            obs,
                            prevact,
                            training=False,
                        )[1][2]["losses"][key].mean()

                    grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
                    metrics[f"gradnorm/{key}"] = optax.global_norm(grad)
                except KeyError:
                    print(f"Skipping gradnorm summary for missing loss: {key}")

        def firsthalf(xs):
            return jax.tree.map(lambda x: x[:report_batch, : tlen // 2], xs)

        def secondhalf(xs):
            return jax.tree.map(lambda x: x[:report_batch, tlen // 2 :], xs)

        dyn_carry = jax.tree.map(lambda x: x[:report_batch], dyn_carry)
        dec_carry = jax.tree.map(lambda x: x[:report_batch], dec_carry)
        dyn_carry, _, obsfeat = self.dyn.observe(
            dyn_carry,
            firsthalf(outs["tokens"]),
            firsthalf(prevact),
            firsthalf(obs["is_first"]),
            training=False,
        )
        _, imgfeat, _ = self.dyn.imagine(
            dyn_carry,
            secondhalf(prevact),
            length=tlen - tlen // 2,
            training=False,
        )
        dec_carry, _, obsrecons = self.dec(
            dec_carry,
            obsfeat,
            firsthalf(obs["is_first"]),
            training=False,
        )
        dec_carry, _, imgrecons = self.dec(
            dec_carry,
            imgfeat,
            jnp.zeros_like(secondhalf(obs["is_first"])),
            training=False,
        )

        for key in self.dec.imgkeys:
            assert obs[key].dtype == jnp.uint8
            true = obs[key][:report_batch]
            pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
            pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
            error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
            video = jnp.concatenate([true, pred, error], 2)

            video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
            mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
            border = jnp.full((tlen, 3), jnp.array([0, 255, 0]), jnp.uint8)
            border = border.at[tlen // 2 :].set(jnp.array([255, 0, 0], jnp.uint8))
            video = jnp.where(mask, video, border[None, :, None, None, :])
            video = jnp.concatenate([video, 0 * video[:, :10]], 1)

            _, tsize, height, width, channels = video.shape
            grid = video.transpose((1, 2, 0, 3, 4)).reshape(
                (tsize, height, report_batch * width, channels)
            )
            metrics[f"openloop/{key}"] = grid

        carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
        del entries
        return carry, metrics

    def _select_encoder_obs(self, obs):
        return self._pick_obs(obs, _VECTOR_KEYS if self.config.enc.typ == "simple" else _GRAPH_KEYS)

    def _select_decoder_obs(self, obs):
        return self._pick_obs(obs, _VECTOR_KEYS)

    @staticmethod
    def _pick_obs(obs, keys):
        return {k: obs[k] for k in keys}

    def _apply_replay_context(self, carry, data):
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        carry = (enc_carry, dyn_carry, dec_carry)
        stepid = data["stepid"]
        obs = {k: data[k] for k in self.obs_space}

        def prepend(x, y):
            return jnp.concatenate([x[:, None], y[:, :-1]], 1)

        prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
        if not self.config.replay_context:
            return carry, obs, prevact, stepid

        replay = elements.tree.nestdict(data)
        k_replay = self.config.replay_context
        entries = [replay.get(k, {}) for k in ("enc", "dyn", "dec")]

        def lhs(xs):
            return jax.tree.map(lambda x: x[:, :k_replay], xs)

        def rhs(xs):
            return jax.tree.map(lambda x: x[:, k_replay:], xs)

        rep_carry = (
            self.enc.truncate(lhs(entries[0]), enc_carry),
            self.dyn.truncate(lhs(entries[1]), dyn_carry),
            self.dec.truncate(lhs(entries[2]), dec_carry),
        )
        rep_obs = {k: rhs(data[k]) for k in self.obs_space}
        rep_prevact = {k: data[k][:, k_replay - 1 : -1] for k in self.act_space}
        rep_stepid = rhs(stepid)

        first_chunk = data["consec"][:, 0] == 0
        carry, obs, prevact, stepid = jax.tree.map(
            lambda normal, replay: nn.where(first_chunk, replay, normal),
            (carry, rhs(obs), rhs(prevact), rhs(stepid)),
            (rep_carry, rep_obs, rep_prevact, rep_stepid),
        )
        return carry, obs, prevact, stepid

    def _make_opt(
        self,
        lr: float = 4e-5,
        agc: float = 0.3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: bool = True,
        nesterov: bool = False,
        wd: float = 0.0,
        wdregex: str = r"/kernel$",
        schedule: str = "const",
        warmup: int = 1000,
        anneal: int = 0,
    ):
        chain = []
        chain.append(embodied.jax.opt.clip_by_agc(agc))
        chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
        chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
        if wd:
            assert not wdregex[0].isnumeric(), wdregex
            pattern = re.compile(wdregex)

            def wdmask(params):
                return {k: bool(pattern.search(k)) for k in params}

            chain.append(optax.add_decayed_weights(wd, wdmask))
        assert anneal > 0 or schedule == "const"
        if schedule == "const":
            sched = optax.constant_schedule(lr)
        elif schedule == "linear":
            sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
        elif schedule == "cosine":
            sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
        else:
            raise NotImplementedError(schedule)
        if warmup:
            ramp = optax.linear_schedule(0.0, lr, warmup)
            sched = optax.join_schedules([ramp, sched], [warmup])
        chain.append(optax.scale_by_learning_rate(sched))
        return optax.chain(*chain)
