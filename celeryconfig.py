"""
Celery configuration for time series forecasting tasks
"""

import os

# Broker settings
broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Task settings
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'UTC'
enable_utc = True

# Worker settings
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 1000
worker_disable_rate_limits = False

# Task routing
task_routes = {
    'forecast_stock': {'queue': 'forecasting'},
    'batch_forecast': {'queue': 'forecasting'},
    'update_historical_data': {'queue': 'data_management'},
    'cleanup_old_forecasts': {'queue': 'maintenance'},
}

# Queue definitions
task_default_queue = 'default'
task_queues = {
    'default': {
        'exchange': 'default',
        'routing_key': 'default',
    },
    'forecasting': {
        'exchange': 'forecasting',
        'routing_key': 'forecasting',
        'queue_arguments': {'x-max-priority': 10},
    },
    'data_management': {
        'exchange': 'data_management',
        'routing_key': 'data_management',
    },
    'maintenance': {
        'exchange': 'maintenance',
        'routing_key': 'maintenance',
    },
}

# Task priority
task_default_priority = 5
task_queue_max_priority = 10

# Result backend settings
result_expires = 3600  # 1 hour
result_persistent = True

# Task execution settings
task_always_eager = False
task_eager_propagates = True
task_ignore_result = False
task_store_errors_even_if_ignored = True

# Logging
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] [%(task_name)s(%(task_id)s)] %(message)s'

# Monitoring
worker_send_task_events = True
task_send_sent_event = True

# Security
broker_connection_retry_on_startup = True
broker_connection_retry = True
broker_connection_max_retries = 10

# Performance
worker_concurrency = os.getenv('CELERY_WORKER_CONCURRENCY', 4)
worker_max_memory_per_child = 200000  # 200MB
worker_max_tasks_per_child = 1000

# Beat schedule (for periodic tasks)
beat_schedule = {
    'update-historical-data-daily': {
        'task': 'update_historical_data',
        'schedule': 86400.0,  # 24 hours
        'args': (),
        'kwargs': {'stock_symbol': 'AAPL'},  # Default stock
    },
    'cleanup-old-forecasts-weekly': {
        'task': 'cleanup_old_forecasts',
        'schedule': 604800.0,  # 7 days
        'args': (),
        'kwargs': {'days_to_keep': 30},
    },
}

# Task time limits
task_soft_time_limit = 300  # 5 minutes
task_time_limit = 600  # 10 minutes

# Result backend timeouts
result_backend_transport_options = {
    'master_name': 'mymaster',
    'visibility_timeout': 3600,
    'fanout_prefix': True,
    'fanout_patterns': True,
}

# Redis specific settings
broker_transport_options = {
    'visibility_timeout': 3600,
    'fanout_prefix': True,
    'fanout_patterns': True,
    'socket_connect_timeout': 5,
    'socket_timeout': 5,
    'retry_on_timeout': True,
}

# Error handling
task_annotations = {
    '*': {
        'rate_limit': '10/m',  # 10 tasks per minute
        'time_limit': 600,     # 10 minutes
        'soft_time_limit': 300, # 5 minutes
    },
    'forecast_stock': {
        'rate_limit': '5/m',   # 5 forecasts per minute
        'time_limit': 1800,    # 30 minutes
        'soft_time_limit': 900, # 15 minutes
    },
    'batch_forecast': {
        'rate_limit': '2/m',   # 2 batch operations per minute
        'time_limit': 3600,    # 1 hour
        'soft_time_limit': 1800, # 30 minutes
    },
}
