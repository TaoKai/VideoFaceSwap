"""
日志模块
"""
import logging
from logging import handlers
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
logger_dir = os.path.join(base_dir, 'logs')
# print(logger_dir)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not os.path.exists(logger_dir):
    os.makedirs(logger_dir)
log_file = os.path.join(logger_dir, 'daily')
# if not os.path.exists(log_file):
#     with open(log_file, 'w'): pass
# fh = handlers.RotatingFileHandler(log_file, maxBytes=10737418240, backupCount=5)    # 10G
# fh.setLevel(logging.INFO)

th = handlers.TimedRotatingFileHandler(log_file, when='D', backupCount=7, encoding='utf-8')
# th.suffix = "%Y%m%d-%H%M.log"
th.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
th.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(th)
logger.addHandler(ch)
