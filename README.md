# Install
```bash
us sync
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
- `<algo>`: tqn or ppo or trpo

# example
bash train.sh train tqn
```
validate:
```bash
bash train.sh val tqn
```