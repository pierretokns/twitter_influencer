# Cloudflare Tunnel Setup

This folder contains configuration for exposing the LinkedIn tools via Cloudflare Tunnel.

## Why Tunnel Instead of Pages?

Cloudflare Pages is for static sites only. For our Flask apps, we need either:
- **DNS A record** (exposes raw IP, no proxying)
- **Cloudflare Tunnel** (recommended - secure, proxied, no exposed ports)

## Setup on Server

### 1. Install cloudflared

```bash
# On the Hetzner server
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

### 2. Authenticate

```bash
cloudflared tunnel login
# Opens browser to authenticate with your Cloudflare account
```

### 3. Create Tunnel

```bash
cloudflared tunnel create linkedin-tools
# Note the tunnel ID output
```

### 4. Configure Tunnel

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: /home/appuser/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  # Tournament Dashboard
  - hostname: tournament.yourdomain.com
    service: http://localhost:5001

  # LinkedIn Feed
  - hostname: feed.yourdomain.com
    service: http://localhost:5002

  # Catch-all (required)
  - service: http_status:404
```

### 5. Create DNS Records

```bash
cloudflared tunnel route dns linkedin-tools tournament.yourdomain.com
cloudflared tunnel route dns linkedin-tools feed.yourdomain.com
```

### 6. Run as Service

```bash
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

## Alternative: Simple DNS A Record

If you prefer exposing ports directly:

1. Log into Cloudflare Dashboard
2. Go to DNS for your domain
3. Add A records:
   - `tournament.yourdomain.com` -> `157.90.125.102` (Proxy disabled for non-standard ports)
   - `feed.yourdomain.com` -> `157.90.125.102`

Note: Cloudflare proxy only works on standard ports (80, 443). For ports 5001/5002,
you'd need to either:
- Use Tunnel (recommended)
- Disable proxy (orange cloud off) and access via `domain:port`
- Set up nginx reverse proxy on port 80/443

## Files in This Folder

- `config.yml.example` - Example tunnel configuration (copy to server)
- Actual credentials and config are not committed (in .gitignore)
