# Diffusion Tuning

Diffusion Tuning: Transferring Diffusion Models via Chain of Forgetting (NeurIPS 2024) [Paper Link](https://openreview.net/forum?id=S98OzJD3jn)

This repository contains a minimal implementation of Diffusion Tuning, adapted from the [DiT repository](https://github.com/facebookresearch/DiT).

## Implementation

To help the community integrate the spirit of Diffusion Tuning with their own codebase, we present the core snapshot in this section.

> Core Snapshot #1: defining the sampling probs of `t` and `t_s`.

```python

probs = torch.Tensor([i**args.prob_scale for i in range (1,1001)]) # sampling probs
rev_probs = torch.Tensor([i**args.LWF_prob_scale for i in range (1000,0,-1)]) 
probs = probs / probs.sum()
rev_probs = rev_probs / rev_probs.sum()

categorical_dist = torch.distributions.categorical.Categorical(probs=probs)
rev_categorical_dist = torch.distributions.categorical.Categorical(probs=rev_probs)

######

t = categorical_dist.sample((x.shape[0],)).to(device)
t_s = rev_categorical_dist.sample((x_s.shape[0],)).to(device)

```

> Core Snapshot #2: As the label space of the fine-tuning tasks may differ from the pre-trained task. For all generated images, we only train the unconditional branch.
> 
```python

y_embed_split = [x.size(0), x_s.size(0)]
x = torch.cat([x, x_s], dim=0)
y = torch.cat([y, y_s], dim=0)
t = torch.cat([t, t_s], dim=0)

###

y_t, y_s = torch.split(y, y_embed_split)
dummy_y = torch.tensor([1000] * y_s.size(0), device=y.device, dtype=torch.long ) # hand-craft-drop for training
y = torch.cat([y_t,dummy_y], dim=0)

```

The full code can be check in the file.

## BibTeX
If you find that our work is useful for you, please add the following citation.

```bibtex
@inproceedings{
zhong2024diffusion,
title={Diffusion Tuning: Transferring Diffusion Models via Chain of Forgetting},
author={Jincheng Zhong and Xingzhuo Guo and Jiaxiang Dong and Mingsheng Long},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=S98OzJD3jn}
}
```

## Contact

If you have any question, please contact zhongjinchengwork@gmail.com.