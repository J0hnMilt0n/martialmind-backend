# 🚀 MartialMind AI Backend Deployment Guide

## Quick Deployment Options

### Option 1: Railway (Recommended for MVP)

**Pros:** Easy, fast, automatic HTTPS, built-in monitoring  
**Cost:** ~$5-10/month

**Steps:**

1. **Install Railway CLI:**
```bash
npm install -g @railway/cli
```

2. **Login:**
```bash
railway login
```

3. **Initialize project:**
```bash
cd martialmind-backend
railway init
```

4. **Deploy:**
```bash
railway up
```

5. **Set environment variables:**
```bash
railway variables set POSE_MODEL_COMPLEXITY=1
railway variables set MIN_DETECTION_CONFIDENCE=0.5
```

6. **Get deployment URL:**
```bash
railway domain
```

7. **Update Dojo Republic .env:**
```env
MARTIALMIND_API_URL=https://your-railway-app.railway.app
```

---

### Option 2: Render

**Pros:** Good for Python apps, generous free tier  
**Cost:** Free tier available, $7/month for paid

**Steps:**

1. **Push code to GitHub**

2. **Create New Web Service on Render:**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure service:**
   - **Name:** martialmind-ai
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Add environment variables:**
   - POSE_MODEL_COMPLEXITY=1
   - MIN_DETECTION_CONFIDENCE=0.5
   - (add all from .env.example)

5. **Deploy** and copy the URL

6. **Update Dojo Republic .env:**
```env
MARTIALMIND_API_URL=https://martialmind-ai.onrender.com
```

---

### Option 3: AWS EC2 (Production-Grade)

**Pros:** Full control, scalable, professional  
**Cost:** ~$10-50/month depending on instance

**Steps:**

1. **Launch EC2 Instance:**
   - Choose Ubuntu 22.04 LTS
   - Instance type: t3.medium (2 vCPU, 4GB RAM)
   - Configure security group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS)

2. **Connect to instance:**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

3. **Install dependencies:**
```bash
sudo apt update
sudo apt install -y python3.9 python3-pip python3-venv nginx
```

4. **Setup application:**
```bash
# Clone repository
git clone https://github.com/your-username/dojo-republic.git
cd dojo-republic/martialmind-backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn
```

5. **Create systemd service:**
```bash
sudo nano /etc/systemd/system/martialmind.service
```

Add:
```ini
[Unit]
Description=MartialMind AI API
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/dojo-republic/martialmind-backend
Environment="PATH=/home/ubuntu/dojo-republic/martialmind-backend/venv/bin"
ExecStart=/home/ubuntu/dojo-republic/martialmind-backend/venv/bin/gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 main:app

[Install]
WantedBy=multi-user.target
```

6. **Start service:**
```bash
sudo systemctl start martialmind
sudo systemctl enable martialmind
sudo systemctl status martialmind
```

7. **Configure Nginx reverse proxy:**
```bash
sudo nano /etc/nginx/sites-available/martialmind
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/martialmind /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

8. **Setup SSL with Let's Encrypt:**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

9. **Update Dojo Republic .env:**
```env
MARTIALMIND_API_URL=https://your-domain.com
```

---

### Option 4: DigitalOcean App Platform

**Pros:** Simple, Docker support, good pricing  
**Cost:** $5-12/month

**Steps:**

1. **Create digitalocean.yaml:**
```yaml
name: martialmind-ai
services:
  - name: api
    source_dir: /martialmind-backend
    build_command: pip install -r requirements.txt
    run_command: uvicorn main:app --host 0.0.0.0 --port 8080
    environment_slug: python
    http_port: 8080
    instance_size_slug: basic-xs
    envs:
      - key: POSE_MODEL_COMPLEXITY
        value: "1"
      - key: MIN_DETECTION_CONFIDENCE
        value: "0.5"
```

2. **Deploy via CLI or Dashboard**

3. **Update Dojo Republic:**
```env
MARTIALMIND_API_URL=https://martialmind-ai-xxxxx.ondigitalocean.app
```

---

### Option 5: Docker Deployment

**For any platform supporting Docker**

1. **Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Build and run:**
```bash
docker build -t martialmind-ai .
docker run -p 8000:8000 -e POSE_MODEL_COMPLEXITY=1 martialmind-ai
```

3. **Deploy to any container platform (Heroku, Fly.io, etc.)**

---

## Frontend Integration

After deploying backend, update Dojo Republic:

**File: `.env.local`**
```env
MARTIALMIND_API_URL=https://your-backend-url.com
```

**Restart Next.js:**
```bash
npm run dev  # Development
# or
npm run build && npm start  # Production
```

---

## Testing Deployment

```bash
# Health check
curl https://your-backend-url.com/health

# Test endpoint
curl https://your-backend-url.com/test

# Test video upload (requires video file)
curl -X POST -F "video=@test-video.mp4" https://your-backend-url.com/analyze-video
```

---

## Performance Optimization

### For Production:

1. **Enable Caching:**
   - Use Redis for repeated analysis caching
   - Cache pose estimations

2. **Use GPU Acceleration (if available):**
   - AWS EC2 with GPU (p3 instances)
   - GCP with GPU support
   - Reduces processing time by 3-5x

3. **Implement Job Queue:**
   - Use Celery + Redis for async processing
   - Return job ID immediately
   - Poll for results

4. **Video Optimization:**
   - Limit video length to 30 seconds
   - Downsample to 720p
   - Process every 2-3 frames

---

## Monitoring & Logging

### Railway/Render:
- Built-in logs and metrics
- Check dashboard for errors

### AWS/Self-hosted:
```bash
# View logs
sudo journalctl -u martialmind -f

# Monitor resources
htop
```

### Add logging to main.py:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

---

## Cost Estimates

| Platform | Free Tier | Paid Plan | Best For |
|----------|-----------|-----------|----------|
| Railway | 500 hours | $5-10/mo | MVP, Quick Start |
| Render | 750 hours | $7/mo | Testing, Development |
| DigitalOcean | No | $5-12/mo | Production |
| AWS EC2 | 750 hours (1 year) | $10-50/mo | Scaling, Professional |

---

## Troubleshooting

### "Module not found" errors:
```bash
pip install -r requirements.txt --force-reinstall
```

### "Out of memory" errors:
- Reduce VIDEO_MAX_FRAMES
- Increase instance RAM
- Process fewer frames

### "Video processing timeout":
- Increase timeout in API route
- Implement async processing
- Use background jobs

### "Cannot connect to backend":
- Check CORS settings
- Verify environment variable
- Check firewall rules

---

## Security Checklist

- [ ] Enable HTTPS/SSL
- [ ] Implement rate limiting
- [ ] Add API key authentication
- [ ] Validate file uploads
- [ ] Delete videos after processing
- [ ] Set max file size limits
- [ ] Use environment variables for secrets
- [ ] Enable CORS only for your domain

---

## Next Steps After Deployment

1. **Test thoroughly** with various videos
2. **Monitor performance** and errors
3. **Collect user feedback**
4. **Iterate on algorithms**
5. **Add analytics** (track usage, success rates)

---

**You're now ready to deploy MartialMind AI! 🚀**

Choose the option that best fits your needs and budget. Railway is recommended for quick MVP deployment.
