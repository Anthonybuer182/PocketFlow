import logging
import sys
import os


# logger.py 内容更新如下
def get_logger(logger_name: str, level: int = logging.DEBUG, 
               formatter: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
               log_file: str = None) -> logging.Logger:
    """
    获取一个配置正确的日志记录器，解决乱码问题
    
    Args:
        logger_name (str): 日志记录器名称
        level (int): 日志级别，默认DEBUG
        formatter (str): 日志格式
        log_file (str): 可选的日志文件路径
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(logger_name)
    
    # 避免重复添加处理器
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(level)
    log_formatter = logging.Formatter(formatter)
    
    # 控制台处理器 - 强制UTF-8编码
    class UTF8StreamHandler(logging.StreamHandler):
        """自定义UTF-8编码的流处理器"""
        def __init__(self, stream=None):
            super().__init__(stream)
            self.encoding = 'utf-8'
        
        def emit(self, record):
            try:
                msg = self.format(record)
                if not isinstance(msg, str):
                    msg = str(msg)
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器 - 强制UTF-8编码
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    
    # 防止日志传播到根记录器
    logger.propagate = False
    
    return logger
