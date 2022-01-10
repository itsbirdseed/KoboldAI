'''
Some of the code in this file is from VE FORBRYDERNE's fork of Mesh Transformer JAX:
https://github.com/VE-FORBRYDERNE/mesh-transformer-jax/tree/ck
That project is Apache 2.0 licensed; this file is still AGPL-3.0.

                                 Apache License
                           Version 2.0, January 2004
                        https://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   Copyright 2021 Ben Wang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''


import multiprocessing
from typing import Any, Dict, List, Optional
import progressbar
import time
import os
import requests
import random
import jax
from jax.config import config
from jax.experimental import maps
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import transformers
from mesh_transformer.checkpoint import read_ckpt_lowmem
from mesh_transformer.transformer_shard import CausalTransformer, CausalTransformerShard
from mesh_transformer.layers import fixed_pos_embedding, apply_rotary_pos_emb
from mesh_transformer.util import g_psum, f_psum


params: Dict[str, Any] = {}


def show_spinner():
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, widgets=[progressbar.Timer(), '  ', progressbar.BouncingBar(left='[', right=']', marker='█')])
    i = 0
    while True:
        bar.update(i)
        time.sleep(0.1)
        i += 1

def apply_repetition_penalty(logits, tokens, repetition_penalty):
    '''
    This gets called by generate_loop_fn to apply repetition penalty
    to the 1D array logits using the provided 1D array of tokens to penalize
    '''
    # Make a new array with the same length as the tokens array but with
    # each element replaced by the value at the corresponding index in the
    # logits array; e.g.
    # if logits is [77, 5, 3, 98] and tokens is [0, 1, 2, 3, 2, 3, 1],
    # then penalty_logits will be [77, 5, 3, 98, 3, 98, 5]
    penalty_logits = jnp.take(logits, tokens)
    # Divide positive values by repetition_penalty and multiply negative
    # values by repetition_penalty (the academic publication that described
    # this technique actually just only divided, but that would cause tokens
    # with negative logits to become more likely, which is obviously wrong)
    penalty_logits = jnp.where(
        penalty_logits > 0,
        penalty_logits/repetition_penalty,
        penalty_logits*repetition_penalty,
    )
    # Finally, put those penalized logit values back into their original
    # positions in the logits array
    return logits.at[tokens].set(penalty_logits)

def kobold_sample(key, logits, top_p=0.9, temp=0.5, top_k=0, tfs=1.0):
    '''
    This gets called by generate_loop_fn to apply a series of 4 filters
    to the logits (top-k, then top-p, then TFS, then temperature) before
    picking one token using the modified logits
    '''
    # Top-k (keep only the k tokens with the highest logits and remove
    # the rest, by setting their logits to negative infinity)
    def top_k_filter(logits):
        # After sorting the logits array in descending order,
        # sorted_indices_to_remove is a 1D array that is True for tokens
        # in the sorted logits array we want to remove and False for ones
        # we want to keep, in this case the first top_k elements will be
        # False and the rest will be True
        sorted_indices_to_remove = jnp.arange(len(logits)) >= top_k
        # Unsort the logits array back to its original configuration and
        # remove tokens we need to remove
        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-logits),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, logits)
    logits = jax.lax.cond(top_k > 0, top_k_filter, lambda x: x, logits)
    # Top-p (after sorting the remaining tokens again in descending order of
    # logit, remove the ones that have cumulative softmax probability
    # greater than p)
    def top_p_filter(logits):
        # Sort the logits array in descending order, replace every element
        # with e (Euler's number) to the power of that element, and divide
        # each element of the new array by the sum of the elements in the
        # new array
        sorted_logits = -jnp.sort(-logits)
        probabilities = jax.nn.softmax(sorted_logits)
        # Calculate cumulative_probabilities as the prefix-sum array of
        # probabilities
        cumulative_probabilities = jnp.cumsum(probabilities, axis=-1)
        # We want to remove tokens with cumulative probability higher
        # than top_p
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Don't ever remove the token with the highest logit, even if
        # the probability is higher than top_p
        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
        # Unsort and remove
        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-logits),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, logits)
    logits = jax.lax.cond(top_p < 1.0, top_p_filter, lambda x: x, logits)
    # Tail free sampling (basically top-p a second time on remaining tokens
    # except it's the "cumulative normalized absolute second finite
    # differences of the softmax probabilities" instead of just the
    # cumulative softmax probabilities)
    def tail_free_filter(logits):
        # Sort in descending order
        sorted_logits = -jnp.sort(-logits)
        # Softmax again
        probabilities = jax.nn.softmax(sorted_logits)
        # Calculate the second finite differences of that array (i.e.
        # calculate the difference array and then calculate the difference
        # array of the difference array)
        d2 = jnp.diff(jnp.diff(probabilities))
        # Get the absolute values of all those second finite differences
        d2 = jnp.abs(d2)
        # Normalize (all elements in the array are divided by the sum of the
        # array's elements)
        d2 = d2 / d2.sum(axis=-1, keepdims=True)
        # Get the prefix-sum array
        cumulative_d2 = jnp.cumsum(d2, axis=-1)
        # We will remove the tokens with a cumulative normalized absolute
        # second finite difference larger than the TFS value
        sorted_indices_to_remove = cumulative_d2 > tfs
        # Don't remove the token with the highest logit
        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
        # Since the d2 array has two fewer elements than the logits array,
        # we'll add two extra Trues to the end
        sorted_indices_to_remove = jnp.pad(
            sorted_indices_to_remove,
            (0, 2),
            constant_values=True,
        )
        # Unsort and remove
        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-logits),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, logits)
    logits = jax.lax.cond(tfs < 1.0, tail_free_filter, lambda x: x, logits)
    # Temperature (just divide the logits by the temperature)
    def temp_filter(logits):
        return logits / temp
    logits = jax.lax.cond(True, temp_filter, lambda x: x, logits)
    # Finally, pick one token using the softmax thingy again (it gives
    # an array whose elements sum to 1 so it can be used nicely as a
    # probability distribution)
    return jax.random.categorical(key, logits, -1).astype(jnp.uint32)[jnp.newaxis], None

pad_token_id = 50256


def self_embed(config, x, embed_param, dtype=jnp.bfloat16, pe_length=0, soft_embeddings=None, positional_embeddings=None):
    in_dim = config["n_vocab"] + config.get("n_vocab_padding", 0)
    out_dim = config["d_model"]
    shards = config["cores_per_replica"]
    in_dim_per_shard = in_dim // shards
    out_dim_per_shard = out_dim // shards

    shard_start_index = jax.lax.axis_index('shard') * in_dim_per_shard

    input_onehot = jax.nn.one_hot(x - shard_start_index, in_dim_per_shard)
    proj_out = input_onehot * embed_param  # proj_out = self.proj(input_onehot)

    mask = jnp.broadcast_to((x < in_dim)[:, jnp.newaxis], proj_out.shape)
    proj_out = jnp.where(mask, proj_out, 0)

    if soft_embeddings is not None:
        assert soft_embeddings.ndim == 2
        assert soft_embeddings.shape[1] == out_dim

        soft_shard_start_index = in_dim + jax.lax.axis_index('shard') * soft_embeddings.shape[0]

        input_soft_onehot = jax.nn.one_hot(x - soft_shard_start_index, soft_embeddings.shape[0])
        proj_out += jnp.dot(input_soft_onehot, soft_embeddings)

    proj_out = g_psum(proj_out)

    if positional_embeddings is not None:
        pe_length = jnp.int32(pe_length)
        shard_roll_index = jnp.int32(jax.lax.axis_index('shard') * out_dim_per_shard)
        pos_embed = jnp.pad(positional_embeddings, ((0, 0), (0, out_dim - out_dim_per_shard)))
        pos_embed = jnp.roll(pos_embed, shard_roll_index, axis=1)
        pos_embed = jnp.roll(pos_embed, -pe_length, axis=0)[-proj_out.shape[0]:]
        proj_out += pos_embed

    proj_out = g_psum(proj_out)
    return proj_out


def self_attn(config, q, v, k, attn_bias, o_param):
    heads = config["n_heads"]
    dim = config["d_model"]
    shards = config["cores_per_replica"]
    dim_per_head = dim // heads
    dim_per_shard = dim // shards
    is_rotary = config["pe"] == "rotary"
    pe_rotary_dims = config.get("pe_rotary_dims", dim_per_head)
    compat = config.get("compat", "j")

    if is_rotary:
        k_rot = k[:, :, :pe_rotary_dims]
        k_pass = k[:, :, pe_rotary_dims:]

        q_rot = q[:, :, :pe_rotary_dims]
        q_pass = q[:, :, pe_rotary_dims:]

        sincos = fixed_pos_embedding(k_rot)
        q_rot = apply_rotary_pos_emb(q_rot, sincos)
        k_rot = apply_rotary_pos_emb(k_rot, sincos)

        k = jnp.concatenate([k_rot, k_pass], axis=-1)
        q = jnp.concatenate([q_rot, q_pass], axis=-1)

    attention_logits = jnp.einsum("thd,Thd->htT", q, k)

    if compat != "neo":
        sqrt_key_size = jnp.sqrt(dim_per_head).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

    attention_logits += attn_bias

    attention_weights = jax.nn.softmax(attention_logits)
    attention_vec = jnp.einsum("htT,Thd->thd", attention_weights, v).reshape((-1, dim_per_shard))

    return attention_vec * o_param  # return self.o(attention_vec)


def self_norm(inputs, norm_scale_param, norm_offset_param):
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    variance = jnp.var(inputs, axis=-1, keepdims=True)

    param_shape = inputs.shape[-1:]
    scale = norm_scale_param  # scale = hk.get_parameter("scale", param_shape, inputs.dtype, init=jnp.ones)
    scale = jax.lax.all_gather(scale, "shard")[0]

    offset = norm_offset_param  # offset = hk.get_parameter("offset", param_shape, inputs.dtype, init=jnp.zeros)
    offset = jax.lax.all_gather(offset, "shard")[0]

    scale = jnp.broadcast_to(scale, inputs.shape)
    offset = jnp.broadcast_to(offset, inputs.shape)
    mean = jnp.broadcast_to(mean, inputs.shape)

    inv = scale * jax.lax.rsqrt(variance + 1e-5)
    if True:  # if self.offset:
        return inv * (inputs - mean) + offset
    else:
        return inv * (inputs - mean)


def l_ff(x, dense_proj_param, dense_proj_o_param):
    dense_proj = x * dense_proj_param  # dense_proj = self.dense_proj(x)
    dense_proj = jax.nn.gelu(dense_proj)
    return dense_proj * dense_proj_o_param  # return self.dense_proj_o(dense_proj)


def l_neo_ff(x, dense_proj_param, dense_proj_o_param, norm_scale_param, norm_offset_param):
    x = self_norm(x, norm_scale_param, norm_offset_param)  # x = self.norm_2(x)
    dense_out = l_ff(x, dense_proj_param, dense_proj_o_param)
    return g_psum(dense_out)


def l_decode_once(config, attention_type, decode_state, x, attn_bias, q_param, k_param, v_param, o_param, norm_scale_param, norm_offset_param):
    heads = config["n_heads"]
    dim = config["d_model"]
    shards = config["cores_per_replica"]
    dim_per_head = dim // heads
    heads_per_shard = heads // shards
    local_attention_window = config.get("local_attention_window", 256)
    compat = config.get("compat", "j")

    x = f_psum(x)
    x = self_norm(x, norm_scale_param, norm_offset_param)  # x = self.norm(x)

    assert x.shape[0] == 1

    q = (x*q_param).reshape(x.shape[:-1] + (heads_per_shard, dim_per_head))
    v = (x*v_param).reshape(x.shape[:-1] + (heads_per_shard, dim_per_head))
    k = (x*k_param).reshape(x.shape[:-1] + (heads_per_shard, dim_per_head))

    # add new kv to end
    v = jnp.concatenate((decode_state["v"], v), axis=0)[1:]
    k = jnp.concatenate((decode_state["k"], k), axis=0)[1:]

    tokens_decoded = decode_state["tokens_decoded"] + 1
    length = v.shape[0]

    if attention_type == "local":
        masked_tokens = length - jnp.minimum(tokens_decoded, local_attention_window)
    else:
        masked_tokens = length - tokens_decoded

    attention_mask = jnp.arange(0, length) < masked_tokens
    bias = (-1e10 * attention_mask)
    bias += attn_bias

    attn_out = self_attn(q, v, k, bias, o_param)
    if compat == "neo":
        out = attn_out
    else:
        dense_out = l_ff(x)
        out = attn_out + dense_out

    return g_psum(out), {
        "tokens_decoded": tokens_decoded,
        "k": k,
        "v": v
    }


def self_proj(config, x, proj_param, norm_scale_param, norm_offset_param):
    out_dim_unpadded = config["n_vocab"]
    compat = config.get("compat", "j")

    x = self_norm(x, norm_scale_param, norm_offset_param)  # x = self.norm(x)
    proj = x * (proj_param.T if compat == "neo" else proj_param)  # proj = self.proj(x, transpose_weights=self.compat == "neo")

    all_proj = jax.lax.all_gather(proj, 'shard')

    # return hk.Flatten()(jnp.transpose(all_proj, (1, 0, 2)))[:, :self.out_dim_unpadded]
    proj_out = jnp.transpose(all_proj, (1, 0, 2))
    return proj_out.reshape((proj_out.shape[0], -1))[:, :out_dim_unpadded]


def generate_once(config, params, new_tok, state, soft_embeddings=None):
    input_len = state[0]["v"].shape[0]

    attn_bias = 0
    # if self.rpe is not None:
    #     attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
    #     attn_bias = attn_bias[:, -1:, :]
    # else:
    #     attn_bias = 0

    x = self_embed(embed_param, new_tok, pe_length=state[0]["tokens_decoded"] + 1, soft_embeddings=soft_embeddings)

    new_states = []

    for l, s in zip(self_transformer_layers, state):
        res, layer_state = l_decode_once(s, x, attn_bias)
        x = x + res
        if l.compat == "neo":
            x = x + l_neo_ff(x)
        new_states.append(layer_state)

    return self_proj(x), new_states


class PenalizingCausalTransformer(CausalTransformer):
    def __init__(self, config):
        # Initialize
        super().__init__(config)
        # These are the tokens that we don't want the AI to ever write
        self.badwords = jnp.array([6880, 50256, 42496, 4613, 17414, 22039, 16410, 27, 29, 38430, 37922, 15913, 24618, 28725, 58, 47175, 36937, 26700, 12878, 16471, 37981, 5218, 29795, 13412, 45160, 3693, 49778, 4211, 20598, 36475, 33409, 44167, 32406, 29847, 29342, 42669, 685, 25787, 7359, 3784, 5320, 33994, 33490, 34516, 43734, 17635, 24293, 9959, 23785, 21737, 28401, 18161, 26358, 32509, 1279, 38155, 18189, 26894, 6927, 14610, 23834, 11037, 14631, 26933, 46904, 22330, 25915, 47934, 38214, 1875, 14692, 41832, 13163, 25970, 29565, 44926, 19841, 37250, 49029, 9609, 44438, 16791, 17816, 30109, 41888, 47527, 42924, 23984, 49074, 33717, 31161, 49082, 30138, 31175, 12240, 14804, 7131, 26076, 33250, 3556, 38381, 36338, 32756, 46581, 17912, 49146])
        def generate_initial(state, key, ctx, ctx_length, aux, sampler_options, soft_embeddings=None):
            gen_length = self.gen_length
            def generate_initial_inner(context, ctx_length, aux):
                # Give the initial context to the transformer
                transformer = CausalTransformerShard(config)
                _, initial_state = transformer.generate_initial(context, ctx_length, soft_embeddings=soft_embeddings)
                # The "generated" array will contain the tokens from the
                # context as well as the tokens picked by the sampler at
                # each stage, padded with a bunch of 50256s, so we know
                # which tokens have to be repetition penalized
                generated = jnp.pad(context, (0, gen_length), constant_values=pad_token_id)  # Let it start off with just the 2048 context tokens, plus gen_length 50256s which will be eventually filled with sampler-chosen tokens
                generated_index = config["seq"]
                # Add that information to generate_scan_fn's starting state
                initial_state = (generated, generated_index) + initial_state
                return initial_state
            generate_fn = hk.transform(generate_initial_inner).apply
            return generate_fn(state["params"], key, ctx, ctx_length, aux)
        self.generate_initial_xmap = jax.experimental.maps.xmap(
            fun=generate_initial,
            in_axes=(
                ["shard", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["shard", ...],
            ),
            out_axes=["batch", "shard", ...],
            axis_resources={'shard': 'mp', 'batch': 'dp'}
        )
        def generate_once(state, key, ctx, ctx_length, aux, sampler_options, soft_embeddings=None):
            kv_shape = (config["seq"] - 1, config["n_heads"] // config["cores_per_replica"], config["d_model"] // config["n_heads"])
            carry = (
                jnp.pad(ctx, (0, self.gen_length), constant_values=pad_token_id),
                jnp.int32(0),
                jnp.empty(1, dtype=jnp.uint32),
                [
                    {
                        "k": jnp.empty(kv_shape, dtype=jnp.float32),
                        "v": jnp.empty(kv_shape, dtype=jnp.float32),
                        "tokens_decoded": jnp.uint32(0),
                    }
                    for _ in range(config["layers"])
                ],
                jnp.empty(2, dtype=jnp.uint32),
            )
            # Get repetition penalty from the arguments
            repetition_penalty = sampler_options.pop('repetition_penalty', None)
            def generate_once_inner(carry):
                transformer = CausalTransformerShard(config)
                # Unpack current generate_scan_fn state
                generated, generated_index, next_token, decode_state, sample_key = carry
                # Get the pseudo-random number generator key that will
                # be used by kobold_sample to randomly pick a token
                sample_key, new_key = jax.random.split(sample_key)
                # Give the context to the model and get the logits it
                # spits out
                # (a 2D array with 1 row and 50400 columns representing
                # how strongly it thinks each of the 50257 tokens in its
                # vocabulary should be appended to the context, followed
                # by 143 apparently useless columns ???)
                logits, new_state = transformer.generate_once(next_token, decode_state, soft_embeddings=soft_embeddings)
                # Verify that logits does indeed have that many rows and
                # columns (if you get an error here, pray for mercy)
                assert logits.shape == (1, config["n_vocab"])
                # Flatten it into a 1D array to make it easier to use
                logits = logits[0]
                # Apply repetition penalty to all tokens that are
                # currently inside the "generated" array
                if repetition_penalty is not None:
                    logits = apply_repetition_penalty(
                        logits,
                        generated,
                        repetition_penalty
                    )
                # Remove any tokens in the badwords list by setting
                # their logits to negative infinity which effectively
                # makes their probabilities of being chosen zero
                logits = logits.at[self.badwords].set(-jnp.inf)
                # Use the sampler (kobold_sample) to pick one token
                # based on the logits array as a 1D array with 1 element
                # (higher logit means higher probability of being
                # picked, non-linearly)
                next_token, sample_info = kobold_sample(
                    sample_key,
                    logits,
                    None,
                    **sampler_options,
                )
                # Remember what token was picked so we can repetition
                # penalize it next time
                generated = generated.at[generated_index].set(next_token[0])
                generated_index += 1
                # self.return_logits isn't used in this program, but
                # for the sake of compatibility...
                if self.return_logits:
                    output = (next_token, sample_info, logits[jnp.newaxis])
                else:
                    output = (next_token, sample_info)
                # Re-pack the current generate_scan_fn's state so we can
                # get back the same variables the next time
                new_carry = (generated, generated_index, next_token, new_state, new_key)
                return new_carry
            generate_fn = hk.transform(generate_once_inner).apply
            outputs = []
            for i in range(4):
                carry = generate_fn(state["params"], key, carry)
                outputs.append(carry[2])
            return None, outputs
        self.generate_once_xmap = jax.experimental.maps.xmap(
            fun=generate_once,
            in_axes=(
                # ["batch", "shard", ...],
                ["shard", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["shard", ...],
            ),
            out_axes=(
                ["batch", "shard", ...],
                ["batch", ...],
            ),
            axis_resources={'shard': 'mp', 'batch': 'dp'},
        )
    def generate(self, ctx, ctx_length, gen_length, sampler_options, return_logits=False, soft_embeddings=None):
        key = hk.PRNGSequence(random.randint(0, 2 ** 60))
        batch_size = ctx.shape[0]
        aux = jnp.zeros((batch_size, gen_length), dtype=jnp.uint32)
        self.gen_length = gen_length
        self.batch_size = batch_size
        self.return_logits = return_logits
        print(-1)
        carry = self.generate_initial_xmap(
            self.state,
            jnp.array(key.take(batch_size)),
            ctx,
            np.array(ctx_length, dtype=np.uint32),
            aux,
            sampler_options,
            soft_embeddings,
        )
        outputs = []
        for _ in range(gen_length):
            print(_)
            _, output = self.generate_once_xmap(
                # carry,
                self.state,
                jnp.array(key.take(batch_size)),
                ctx,
                np.array(ctx_length, dtype=np.uint32),
                aux,
                sampler_options,
                soft_embeddings,
            )
            outputs.append(output[0])
        return np.concatenate(outputs, axis=-1)


def infer(
    context: np.array,
    top_p=0.9,
    temp=0.5,
    top_k=0,
    tfs=1.0,
    repetition_penalty=1.0,
    numseqs=1,
    gen_len=80,
    soft_embeddings: Optional[np.array] = None,
    soft_tokens: Optional[np.array] = None,
) -> List[str]:
    maps.thread_resources.env = thread_resources_env
    total_batch = numseqs
    tokens = context
    if(soft_tokens is not None):
        tokens = np.uint32(np.concatenate((soft_tokens, tokens)))
    provided_ctx = tokens.shape[0]
    pad_amount = seq - provided_ctx
    padded_tokens = np.pad(tokens, ((pad_amount, 0),), constant_values=pad_token_id)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * provided_ctx
    samples = []
    batched_generator_params = {
        "temp": temp * np.ones(total_batch),
        "top_p": top_p * np.ones(total_batch),
        "tfs": tfs * np.ones(total_batch),
        "repetition_penalty": repetition_penalty * np.ones(total_batch),
        "top_k": np.full(total_batch, top_k, dtype=np.uint32)
    }
    output = network.generate(
        batched_tokens,
        length,
        gen_len,
        batched_generator_params,
        soft_embeddings=soft_embeddings,
    )
    print(output)
    print(output.shape)
    decoded_tokens = output[1][0]
    for o in decoded_tokens[:, :, 0]:
        samples.append(o)
    return samples


def load_model(path: str, driver_version="tpu_driver0.1_dev20210607", **kwargs) -> None:
    global thread_resources_env, seq, tokenizer, network, params

    default_params = {
        "compat": "j",
        "layers": 28,
        "d_model": 4096,
        "n_heads": 16,
        "n_vocab": 50400,
        "n_vocab_padding": 0,
        "norm": "layernorm",
        "pe": "rotary",
        "pe_rotary_dims": 64,
        "seq": 2048,
        "cores_per_replica": 8,
    }
    params = kwargs
    for param in default_params:
        if param not in params:
            params[param] = default_params[param]

    print("Connecting to your Colab instance's TPU", flush=True)
    spinner = multiprocessing.Process(target=show_spinner, args=())
    spinner.start()
    colab_tpu_addr = os.environ['COLAB_TPU_ADDR'].split(':')[0]
    url = f'http://{colab_tpu_addr}:8475/requestversion/{driver_version}'
    requests.post(url)
    spinner.terminate()
    print()
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']

    cores_per_replica = params["cores_per_replica"]
    seq = params["seq"]
    params["optimizer"] = optax.scale(0)
    mesh_shape = (1, cores_per_replica)
    devices = np.array(jax.devices()[:cores_per_replica]).reshape(mesh_shape)
    thread_resources_env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))
    maps.thread_resources.env = thread_resources_env
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

    if not path.endswith("/"):
        path += "/"

    network = PenalizingCausalTransformer(params)
    # network.state = read_ckpt_lowmem(network.state, path, devices.shape[1])
    network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))
