# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import verl
import verl.utils.torch_functional as verl_F


def compute_rloo_advantage_return(data: verl.DataProto, response_mask: torch.Tensor, n_samples, config):
    # calculate rloo reward on different reward sources, and sum again
    def masked_rloo(reward_tensor_original, mask_tensor):
        reward_tensor = reward_tensor_original.clone()
        reward_tensor[~mask_tensor] = 0
        for start_pos in range(0, reward_tensor.shape[0], n_samples):
            cur_rewards_mean = torch.cat(
                [reward_tensor[pos : pos + 1][mask_tensor[pos : pos + 1]].mean(dim=0, keepdim=True) for pos in range(start_pos, start_pos + n_samples)],
                dim=0,
            )
            cur_rewards_sum = cur_rewards_mean.sum()
            cur_reward_baseline = cur_rewards_sum / (n_samples - 1)
            reward_tensor[start_pos : start_pos + n_samples][mask_tensor[start_pos : start_pos + n_samples]] = reward_tensor[start_pos : start_pos + n_samples][mask_tensor[start_pos : start_pos + n_samples]] * (n_samples / (n_samples - 1)) - cur_reward_baseline

        return reward_tensor

    reward_tensors = []

    with torch.no_grad():
        if "rm_scores" in data.batch.keys() and config.algorithm.reward_dpo_coef != 0.0:
            reward_tensor = data.batch["rm_scores"]
            reward_mask = response_mask.bool()

            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.reward_dpo_coef)

        if "acc" in data.batch.keys() and config.algorithm.reward_gt_coef != 0.0:
            reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
            reward_mask = torch.zeros_like(response_mask, dtype=torch.bool)

            prompt_ids = data.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(-1)

            reward_mask[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1,
            ] = True
            reward_tensor[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1,
            ] = data.batch["acc"]

            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.reward_gt_coef)

        final_reward_tensor = sum(reward_tensors)

        returns = (final_reward_tensor * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        advantages = returns.clone()
        advantages = verl_F.masked_whiten(advantages, response_mask)

        return advantages, returns


def compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta):
    cur_scores = ((token_level_scores * response_mask).sum(dim=1) * beta).sigmoid()
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc)
    return cur_dpo_loss

def bradley_terry_loss(seq_score, acc, beta):
    # seq_score: [N+1] tensor of real scores
    #   seq_score[0] = the single strong example
    #   seq_score[1:] = the many weak examples
    # print(f"DEBUG BT: bradley_terry_loss inputs - seq_score: {seq_score}, acc: {acc}, beta: {beta}")
    pos_mask = acc > 0.98
    neg_mask = ~pos_mask
    print(f"DEBUG BT: pos_mask has {pos_mask.sum().item()} True values out of {len(pos_mask)} total")

    pos = seq_score[pos_mask]             # scalar
    neg = seq_score[neg_mask]            # shape [N]
    #print(f"DEBUG BT: pos: {pos}, neg: {neg}")
    if pos.numel() == 0 or neg.numel() == 0:
        print(f"DEBUG BT: pos or neg is empty. pos numel: {pos.numel()}, neg numel: {neg.numel()}. Returning 0.0")
        # Or handle as an error / return specific NaN if this case is unexpected and should result in NaN
        return torch.tensor(0.0, device=seq_score.device, dtype=seq_score.dtype) # Ensure dtype matches to avoid issues

    logits = beta * (pos - neg)       # shape [N]
    print(f"DEBUG BT: logits: {logits}")
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"WARNING BT: logits contains NaN/Inf: {logits}")

    # probability each time that pos beats that neg:
    probs = torch.sigmoid(logits)  # in (0,1)
    print(f"DEBUG BT: probs: {probs}")
    if torch.isnan(probs).any():
        print(f"WARNING BT: probs contains NaN: {probs}")
    if torch.any(probs <= 1e-9) or torch.any(probs >= 1 - 1e-9):
        print(f"WARNING BT: probs contains values very close to 0 or 1: {probs}")

    # now compare to 1's since pos is always the winner:
    loss = torch.nn.functional.binary_cross_entropy(probs, torch.ones_like(probs))
    print(f"DEBUG BT: loss: {loss}")
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"ERROR BT: Calculated loss is NaN/Inf: {loss}")
    return loss

def compute_bt_loss(token_level_scores, acc, response_mask, beta):
    print(f"DEBUG compute_bt_loss: inputs - token_level_scores: {token_level_scores.shape}, acc: {acc.shape}, response_mask: {response_mask.shape}, beta: {beta}")
    if torch.isnan(token_level_scores).any(): print("WARNING compute_bt_loss: token_level_scores has NaN")
    if torch.isnan(acc).any(): print("WARNING compute_bt_loss: acc has NaN")
    # collapse to sequence‐level scores
    seq_score = (token_level_scores * response_mask).sum(dim=1)
    print(f"DEBUG compute_bt_loss: seq_score: {seq_score}")
    if torch.isnan(seq_score).any(): print("WARNING compute_bt_loss: seq_score has NaN")
    return bradley_terry_loss(seq_score, acc, beta)


def compute_bt_loss_weighted(token_level_scores, acc, response_mask, beta):
    print(f"DEBUG BTW: compute_bt_loss_weighted inputs - token_level_scores: {token_level_scores.shape}, acc: {acc.shape}, response_mask: {response_mask.shape}, beta: {beta}")
    if torch.isnan(token_level_scores).any(): print("WARNING BTW: token_level_scores has NaN")
    if torch.isnan(acc).any(): print("WARNING BTW: acc has NaN")

    # 1) collapse token scores → one scalar per sequence
    seq_score = (token_level_scores * response_mask).sum(dim=1)  # [B]
    print(f"DEBUG BTW: seq_score: {seq_score}")
    if torch.isnan(seq_score).any(): print("WARNING BTW: seq_score has NaN")


    # 2) split into positives (acc > 0.98) vs. negatives
    pos_mask = acc > 0.98                                      # [B] bool
    neg_mask = ~pos_mask                                       # [B] bool

    # if we don’t have any positives or negatives, no pairwise comparisons
    if pos_mask.sum()==0 or neg_mask.sum()==0:
        return torch.tensor(0.0, device=seq_score.device)

    # 3) gather scores and acc’s
    pos_scores = seq_score[pos_mask]                           # [P]
    neg_scores = seq_score[neg_mask]                           # [N]
    pos_acc    = acc[pos_mask]                                 # [P]
    neg_acc    = acc[neg_mask]                                 # [N]

    # 4) make all P×N pairwise differences
    #    logits_ij = β * (s_pos[i] - s_neg[j])
    logits = beta * (pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0))  # [P,N]
    probs  = torch.sigmoid(logits)                                      # [P,N]

    # 5) compute per-pair weights = how far apart their accuracies are
    #    (bigger gap → heavier penalty if you get it wrong)
    weights = (pos_acc.unsqueeze(1) - neg_acc.unsqueeze(0)).clamp(min=0.0)  # [P,N]

    # 6) now BCE vs. “always 1” (pos should beat neg), weighted by ∆acc
    loss = torch.nn.functional.binary_cross_entropy(probs,
                                  torch.ones_like(probs),
                                  weight=weights)
    print(f"DEBUG BTW: loss: {loss}")
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"ERROR BTW: Calculated loss is NaN/Inf: {loss}")

    return loss



def compute_detach_dpo_loss_rm(token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"):
    # we always assume that the BoN size equals n_samples
    # mode1: use acc as rm
    # mode2: use Q as rm
    print(f"DEBUG: Initial token_level_scores: {token_level_scores}")
    print(f"DEBUG: Initial acc: {acc}")
    print(f"DEBUG: Initial Q_bc: {Q_bc}")
    print(f"DEBUG: Initial acc_bc: {acc_bc}")
    print(f"DEBUG: Initial response_mask: {response_mask}")
    print(f"DEBUG: Initial beta: {beta}")
    print(f"DEBUG: Initial bon_mode: {bon_mode}")

    cur_Q = (token_level_scores * response_mask).sum(dim=1) * beta
    print(f"DEBUG: cur_Q: {cur_Q}")

    other_Q = torch.zeros_like(cur_Q)
    for i in range(token_level_scores.shape[0]):
        Q_chosen = Q_bc[i][acc_bc[i] < acc[i]] if acc[i] > 0 else Q_bc[i][acc_bc[i] > acc[i]]
        if len(Q_chosen) > 0:
            other_Q[i] = Q_chosen.mean() * beta
        else:
            other_Q[i] = 0
    dpo_loss = -torch.log(torch.sigmoid((cur_Q - other_Q) * ((acc > 0).float() * 2 - 1)))
    if bon_mode == "none":
        dpo_loss = dpo_loss.mean()
    else:
        weight = torch.zeros_like(dpo_loss)
        n_samples = acc_bc.shape[1]
        if bon_mode == "bon_rm":
            for i in range(token_level_scores.shape[0]):
                weight[i] = n_samples * torch.pow((Q_bc[i] * beta <= cur_Q[i]).float().mean(), n_samples - 1)
        elif bon_mode == "bon_acc":
            for i in range(token_level_scores.shape[0]):
                weight[i] = n_samples * torch.pow((acc_bc[i] <= acc[i]).float().mean(), n_samples - 1)
        else:
            raise NotImplementedError
        dpo_loss = (dpo_loss * weight).sum()

    return dpo_loss


def compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples):
    dpo_acc = []
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        cur_scores = (token_level_scores[start_id : start_id + n_samples] * response_mask[start_id : start_id + n_samples]).sum(dim=1)

        def get_upper_triangle(tensor_x):
            diff_matrix = tensor_x.unsqueeze(1) - tensor_x.unsqueeze(0)
            upper_tri_indices = torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            return diff_matrix[upper_tri_indices]

        cur_acc_diff = get_upper_triangle(acc[start_id : start_id + n_samples])  # in range [-1,1]
        cur_score_diff = get_upper_triangle(cur_scores)  # in R
        cur_score_prediction = (cur_score_diff > 0).float()  # in [0,1]
        if cur_acc_diff.abs().sum() == 0:
            cur_acc = torch.zeros_like(cur_score_prediction[0]) + 0.5
        else:
            cur_acc = (((cur_score_diff > 0) == (cur_acc_diff > 0)).float() * cur_acc_diff.abs()).sum() / cur_acc_diff.abs().sum()

        dpo_acc.append(cur_acc.unsqueeze(0))

    return torch.cat(dpo_acc, dim=0).mean()


def compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples):
    return (torch.sign((token_level_scores * response_mask).sum(dim=-1)) == torch.sign(acc * 2 - 1)).float().mean()
