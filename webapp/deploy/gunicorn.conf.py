"""
Gunicorn production configuration for CVaR Portfolio Optimizer.

Usage:
    gunicorn -c deploy/gunicorn.conf.py app:app
"""

import multiprocessing

# Server socket
bind = "127.0.0.1:8000"
backlog = 256

# Worker processes
workers = min(multiprocessing.cpu_count() * 2 + 1, 8)
worker_class = "gthread"
threads = 4
timeout = 180          # optimization can take time on large portfolios
graceful_timeout = 30
keepalive = 5

# Logging
accesslog = "/var/log/cvar-optimizer/access.log"
errorlog = "/var/log/cvar-optimizer/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "cvar-optimizer"

# Security
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190
