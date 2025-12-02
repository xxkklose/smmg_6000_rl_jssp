# Install
```bash
uv sync
```
or
```bash
pip install -r requirements.txt
```

# Train or Validate
```bash
bash train.sh <mode> <algo>
Parameters:
- `<mode>`: train or val
- `<algo>`: dqn or ppo or trpo

# example
bash train.sh train dqn
```
validate:
```bash
bash train.sh val dqn
```
