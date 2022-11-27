# irl_gym
Custom OpenAI gym environments

## Install 
From *irl-gym/* , call

```
pip install -e .
```

## Notes:

- Generally assumes all envs accept a dictionary containing environment parameters. 
- Not currently to date w.r.t. step-api. We only return one bool. Will be updating this in the future. We have suppressed warnings, so there may be other Gym standards we are out of compliance with. The use case of this is primarily research, so we aim to deviate to best benefit the algorithms we are using, see sibling repo [Decision Making Sandbox](https://github.com/wvu-irl/dm-sandbox)
- As opposed to ndarrays, use observations for dictionaries to help with user clarity. 