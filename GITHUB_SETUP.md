# Quick GitHub Setup for openmythos-swarm

This guide gets you publishing this to GitHub in 5 minutes.

## Step 1: Create GitHub Repo

1. Go to https://github.com/new
2. Create repo: **openmythos-swarm**
3. Description: "Decentralized, volunteer-driven training for OpenMythos 10B on Mac Silicon"
4. Choose **Public** (so volunteers can see it)
5. Click "Create repository" (do NOT initialize with README)

## Step 2: Push Local Code

```bash
cd c:\Users\steve\openwrt\ ackup\openmythos-swarm

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: federated training framework for Mac Studios"

# Add remote
git remote add origin https://github.com/YOUR-USERNAME/openmythos-swarm.git

# Push to main
git branch -M main
git push -u origin main
```

## Step 3: Add Deployment Instructions

Go to GitHub repo → Settings → Pages and enable GitHub Pages.  
Add link to `DEPLOYMENT.md` in README.

## Step 4: Invite Collaborators

**GitHub:**
1. Go to repo → Settings → Collaborators
2. Add friends with Mac Studios

**Public recruitment (optional):**
- Tweet the repo
- Post on Hacker News
- Add to federated learning communities

## Step 5: Host Master (Choose One)

### Option A: Local Mac (Simplest)

Run master on your always-on Mac Studio:
```bash
python master/server.py --state-dir ~/openmythos-master
```

Workers fetch specs from local shared storage or GitHub raw URLs.

### Option B: Cloud VM (More Available)

Deploy on Fly.io, AWS, or DigitalOcean:
```bash
# See DEPLOYMENT.md for detailed instructions
```

## Step 6: First Volunteer Round

1. Tell friends to clone the repo
2. They run: `python worker/client.py`
3. Workers fetch from master, train, submit
4. Master aggregates and publishes results
5. Next round begins!

## Project Structure (for volunteers)

```
openmythos-swarm/
├── README.md              ← Start here
├── DEPLOYMENT.md          ← Master/worker setup
├── requirements.txt       ← pip install -r requirements.txt
├── setup.sh               ← Quick setup script
├── common/                ← Shared crypto + manifests
├── master/                ← Master coordinator
├── worker/                ← Worker client
└── tests/                 ← Integration tests (run: python tests/test_basic.py)
```

## GitHub Actions (Optional)

To run tests automatically on push:

```bash
# Create .github/workflows/test.yml
mkdir -p .github/workflows
cat > .github/workflows/test.yml <<'EOF'
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python tests/test_basic.py
EOF

git add .github
git commit -m "Add CI pipeline"
git push
```

## Success Metrics

✅ Repo created and live  
✅ Master running somewhere  
✅ First worker successfully completes round  
✅ Results aggregated and published  
✅ Next round spec published  
✅ Multiple volunteers contributing  

## Next: Scale

Once you have ~5 workers:
1. Implement robust aggregation (trimmed mean, outlier removal)
2. Add reputation scoring
3. Deploy second master for failover
4. Move to hierarchical combiners for >20 workers

---

**Questions?** Open an issue in the GitHub repo.

