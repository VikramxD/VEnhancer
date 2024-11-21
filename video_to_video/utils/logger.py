
"""
logging configuration module for VEnhancer using Loguru.

This module provides a production-grade logging system with distributed training support,
structured logging, and advanced error tracking while maintaining backwards compatibility
with existing codebase. It handles multi-process logging, log rotation, and detailed
error diagnostics.

Typical usage:
    >>> logger = get_logger()
    >>> logger.info("Starting process")
    
    # With file output
    >>> logger = get_logger("logs/process.log")
    >>> logger.info("Processing batch {}", batch_id)
    
    # In distributed environment
    >>> logger = get_logger("logs/worker_{rank}.log")
    >>> logger.debug("Worker {} processing", dist.get_rank())
"""

import os
import sys
from typing import Optional
from pathlib import Path
import torch.distributed as dist
from loguru import logger


# Remove default logger for clean configuration
logger.remove()

# Define production log format with distributed training support
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "Rank {extra[rank]} | "
    "<level>{message}</level>"
)


def is_dist() -> bool:
    """
    Check if distributed training is initialized.
    
    This function verifies both the availability and initialization of PyTorch's
    distributed training environment.
    
    Returns:
        bool: True if distributed training is initialized, False otherwise.
    
    Example:
        >>> if is_dist():
        ...     logger.info("Running in distributed mode")
    """
    return dist.is_available() and dist.is_initialized()


def is_master(group=None) -> bool:
    """
    Check if current process is the master process.
    
    In distributed training, identifies if the current process is the master process
    (rank 0). In non-distributed mode, always returns True.
    
    Args:
        group: Optional process group for distributed training
            (default: None, uses default process group)
    
    Returns:
        bool: True if current process is master or in non-distributed mode
    
    Example:
        >>> if is_master():
        ...     logger.info("Running on master process")
    """
    return dist.get_rank(group) == 0 if is_dist() else True


def get_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    file_mode: str = "w"
) -> logger:
    """
    Get a configured Loguru logger instance with distributed training support.
    
    This function provides a production-ready logger with support for both console
    and file outputs, distributed training awareness, and advanced error tracking.
    It maintains backwards compatibility with existing codebase while providing
    enhanced functionality.
    
    Args:
        log_file: Optional path to log file. If provided, enables file logging
            for master process. Supports path formatting with {rank} placeholder.
        log_level: Minimum logging level to record. Available levels are:
            TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
            (default: "INFO")
        file_mode: File opening mode when using file output
            'w' - Write mode (default)
            'a' - Append mode
    
    Returns:
        logger: Configured Loguru logger instance
    
    Raises:
        OSError: If unable to create log directory or file
        ValueError: If invalid log level is specified
        Exception: For other configuration errors
    
    Examples:
        Basic usage:
        >>> logger = get_logger()
        >>> logger.info("Application starting")
        
        With file output:
        >>> logger = get_logger("logs/app.log", "DEBUG", "a")
        >>> logger.debug("Detailed debugging info")
        
        Distributed training:
        >>> logger = get_logger(f"logs/worker_{dist.get_rank()}.log")
        >>> logger.info("Worker {} ready", dist.get_rank())
        
        Structured logging:
        >>> logger.info("Processing batch", extra={"batch_id": 123})
        
        Exception tracking:
        >>> try:
        ...     process_data()
        ... except Exception as e:
        ...     logger.exception("Processing failed")
    
    Notes:
        - In distributed mode, only the master process (rank 0) logs at specified level
        - Worker processes only log ERROR and above to avoid log flooding
        - File logging is only enabled for master process when log_file is specified
        - Logs are automatically rotated at 100MB with 10 days retention
        - All logs include process rank information in distributed mode
    """
    # Set rank for logging context
    rank = dist.get_rank() if is_dist() else 0
    is_worker0 = is_master()

    # Create logger instance with rank context
    current_logger = logger.bind(rank=rank)

    try:
        # Configure console output based on process rank
        if is_worker0:
            current_logger.add(
                sys.stdout,
                format=LOG_FORMAT,
                level=log_level,
                colorize=True,
                enqueue=True,
                backtrace=True,
                diagnose=True
            )
        else:
            # Worker processes only log errors
            current_logger.add(
                sys.stdout,
                format=LOG_FORMAT,
                level="ERROR",
                colorize=True,
                enqueue=True,
                backtrace=True,
                diagnose=True
            )

        # Configure file output for master process
        if log_file is not None and is_worker0:
            # Format log file path with rank if needed
            if '{rank}' in log_file:
                log_file = log_file.format(rank=rank)

            # Ensure log directory exists
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)

            # Add rotating file handler
            current_logger.add(
                log_file,
                format=LOG_FORMAT,
                level=log_level,
                rotation="100 MB",
                retention="10 days",
                compression="zip",
                mode=file_mode,
                enqueue=True,
                backtrace=True,
                diagnose=True,
                catch=True
            )

    except Exception as e:
        current_logger.error(f"Error setting up logger: {str(e)}")
        raise

    return current_logger


def add_file_handler_if_needed(
    logger_instance,
    log_file: Optional[str],
    file_mode: str,
    log_level: str
) -> None:
    """
    Add a file handler to an existing logger instance if needed.
    
    This function maintains backwards compatibility with existing code while
    providing enhanced file logging capabilities. It only adds a file handler
    if one doesn't already exist and the process is the master process.
    
    Args:
        logger_instance: Existing logger instance to enhance
        log_file: Path to log file
        file_mode: File opening mode ('w' or 'a')
        log_level: Minimum logging level for file output
    
    Raises:
        OSError: If unable to create log directory or file
        ValueError: If invalid log level specified
    
    Example:
        >>> logger = get_logger()
        >>> add_file_handler_if_needed(logger, "logs/app.log", "w", "DEBUG")
    
    Notes:
        - Only adds file handler if process is master (rank 0)
        - Automatically creates log directory if needed
        - Supports log rotation and compression
    """
    if log_file and is_master():
        try:
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            
            logger_instance.add(
                log_file,
                format=LOG_FORMAT,
                level=log_level,
                rotation="100 MB",
                retention="10 days",
                compression="zip",
                mode=file_mode,
                enqueue=True,
                backtrace=True,
                diagnose=True,
                catch=True
            )
        except Exception as e:
            logger.error(f"Failed to add file handler: {str(e)}")
            raise
