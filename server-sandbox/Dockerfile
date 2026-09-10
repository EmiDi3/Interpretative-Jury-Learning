FROM node:20-slim

# Install sandbox dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bubblewrap \
      socat \
      git \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Create non-root user
RUN useradd -m -u 1000 claude

# Deploy managed settings (root-owned, read-only)
RUN mkdir -p /etc/claude-code
COPY managed-settings.json /etc/claude-code/managed-settings.json
RUN chmod 444 /etc/claude-code/managed-settings.json && \
    chown root:root /etc/claude-code/managed-settings.json

# Switch to non-root
USER claude
WORKDIR /home/claude/project

ENTRYPOINT ["claude"]
