# Deployment Guide for OpenMythos Swarm

## GitHub Repo Setup

### 1. Create Repository

```bash
# On GitHub, create a new repo: openmythos-swarm

git init
git add .
git commit -m "Initial commit: federated training framework"
git branch -M main
git remote add origin https://github.com/YOUR-ORG/openmythos-swarm.git
git push -u origin main
```

### 2. Add Collaborators

Invite volunteers (Mac owners) as collaborators or let them fork.

## Master Node Deployment

### Local Mac Studio (Always-On)

```bash
# Clone repo
git clone https://github.com/YOUR-ORG/openmythos-swarm.git
cd openmythos-swarm

# Install
pip install -r requirements.txt

# Start master (runs forever, publishes rounds)
python master/server.py --state-dir ~/openmythos-master
```

State will be saved to `~/openmythos-master/`:
- `keys/master_private.pem` — master's signing key (KEEP SECRET)
- `rounds/round_XXXX/spec.json` — published round specs
- `submissions/round_XXXX/*.json` — worker submissions

### Cloud Deployment (Optional)

If you want to run master on a managed VM:

**Fly.io:**
```bash
# Create Dockerfile
cat > Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "master/server.py", "--state-dir", "/data/master"]
EOF

# Deploy
fly launch --name openmythos-swarm-master
fly deploy

# Bind persistent storage
fly volumes create master_data --size 10
```

**AWS Lambda + S3 (Serverless aggregation):**
```bash
# Publish round every N hours
# Fetch submissions from S3
# Aggregate + publish results
# (More complex; use if you don't want always-on master)
```

## Worker Node Deployment

### Mac Studio Volunteer

```bash
# Clone repo
git clone https://github.com/YOUR-ORG/openmythos-swarm.git
cd openmythos-swarm

# Install
pip install -r requirements.txt

# Create ~/.openmythos-swarm/config.json
mkdir -p ~/.openmythos-swarm
cat > ~/.openmythos-swarm/config.json <<'EOF'
{
  "worker_id": "mac_studio_myname",
  "master_url": "https://raw.githubusercontent.com/YOUR-ORG/openmythos-swarm/main",
  "local_dir": "~/.openmythos-swarm/local"
}
EOF

# Run worker
python worker/client.py --config ~/.openmythos-swarm/config.json
```

### Scheduled Training (Cron)

To contribute automatically during off-peak hours:

```bash
# Edit crontab
crontab -e

# Add (runs every day at 11 PM)
0 23 * * * cd ~/openmythos-swarm && python worker/client.py --round 1
```

### LaunchAgent (macOS Background Service)

```bash
# Create ~/Library/LaunchAgents/com.openmythos.worker.plist
cat > ~/Library/LaunchAgents/com.openmythos.worker.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.openmythos.worker</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/Users/YOUR-USERNAME/openmythos-swarm/worker/client.py</string>
    </array>
    
    <key>StandardOutPath</key>
    <string>/tmp/openmythos-worker.log</string>
    
    <key>StandardErrorPath</key>
    <string>/tmp/openmythos-worker-error.log</string>
    
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

# Load service
launchctl load ~/Library/LaunchAgents/com.openmythos.worker.plist

# Check status
launchctl list | grep openmythos

# Stop service
launchctl unload ~/Library/LaunchAgents/com.openmythos.worker.plist
```

## Publishing Results

Master publishes all round specs and results as JSON. You can sync to GitHub, S3, or IPFS:

### GitHub Sync (Simple)

```bash
# In master node directory
cd master_state
git add .
git commit -m "Round X results"
git push
```

### S3 Sync (Production)

```bash
# Configure AWS credentials
aws configure

# Sync rounds to S3
aws s3 sync master_state/rounds s3://my-bucket/openmythos-rounds
aws s3 sync master_state/submissions s3://my-bucket/openmythos-submissions

# Make public (if desired)
aws s3api put-object-acl --bucket my-bucket --key openmythos-rounds/* --acl public-read
```

### IPFS Pinning

```bash
# Pin round manifests
ipfs add -r master_state/rounds

# Output hash to GitHub README or pinning service
```

## Monitoring

### Master Health

```bash
# Check latest round
tail -n 20 ~/openmythos-master/rounds/round_*/spec.json

# Check submissions
ls -la ~/openmythos-master/submissions/round_*/

# Monitor logs
tail -f ~/openmythos-master/master.log
```

### Worker Health

```bash
# Check local training
tail -f ~/.openmythos-swarm/worker.log

# Verify submission
cat ~/.openmythos-swarm/local/round_1/submission.json
```

## Troubleshooting

### Master can't find submissions

Check that workers are writing to the master's submissions directory:
```bash
find ~/openmythos-master/submissions -type f
```

### Worker can't fetch round spec

Verify master public key is accessible and matches:
```bash
# On master
cat ~/openmythos-master/keys/master_public.pem

# On worker (should match)
cat ~/.openmythos-swarm/master_public.pem
```

### Signature verification failed

Ensure both master and workers use the same signing key:
```bash
# Copy master's public key to workers
scp ~/openmythos-master/keys/master_public.pem worker:/home/user/.openmythos-swarm/
```

## Next Steps

1. **Test locally** with master + 3 workers (this repo includes test suite)
2. **Deploy master** to always-on Mac or cloud VM
3. **Publish repo** to GitHub
4. **Recruit volunteers** (friends with Mac Studios)
5. **Iterate** on aggregation, monitoring, incentives

## Security Checklist

- [ ] Master private key is backed up securely
- [ ] Master public key distributed to all workers
- [ ] HTTPS/TLS configured if using cloud deployment
- [ ] Worker submissions validated before aggregation
- [ ] Outlier detection enabled in aggregation
- [ ] Audit logs enabled for all round decisions
- [ ] Periodic re-rotation of master keys

## References

- [Federated Learning: Communication-Efficient Learning](https://arxiv.org/abs/1602.05629)
- [How to Backdoor Federated Learning](https://arxiv.org/abs/1807.00459) — Why we need validation
- [Flower: A Federated Learning Framework](https://flower.ai/)

