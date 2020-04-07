import logging
from logging import handlers


def easylearn_logger(out_name=None, debug_message="", info_message="", warning_message="", error_message="", critical_message=""):
    #初始化logger
    logger = logging.getLogger()

    #设置日志记录级别
    logger.setLevel(logging.INFO)

    #fmt设置日志输出格式,datefmt设置 asctime 的时间格式
    formatter = logging.Formatter(fmt='[%(asctime)s]%(levelname)s:%(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    #配置日志输出到控制台
    # console = logging.StreamHandler()
    # console.setLevel(logging.WARNING)
    # console.setFormatter(formatter)
    # logger.addHandler(console)

    #配置日志输出到文件

    file_logging = logging.FileHandler(out_name)
    file_logging.setLevel(logging.INFO)
    file_logging.setFormatter(formatter)
    logger.addHandler(file_logging)

    #配置日志输出到文件,限制单个日志文件的最大体积
    # file_rotating_file = handlers.RotatingFileHandler('app_rotating.log', maxBytes=1024, backupCount=3)
    # file_rotating_file.setLevel(logging.WARNING)
    # file_rotating_file.setFormatter(formatter)
    # logger.addHandler(file_rotating_file)

    #配置日志输出到文件,在固定的时间内记录日志文件
    # file_time_rotating = handlers.TimedRotatingFileHandler("app_time.log", when="s", interval=10, backupCount=5)
    # file_time_rotating.setLevel(logging.INFO)
    # file_time_rotating.setFormatter(formatter)
    # logger.addHandler(file_time_rotating)

    #use
    logger.debug(debug_message)
    logger.info(info_message)
    logger.warning(warning_message)
    logger.error(error_message)
    logger.critical(critical_message)


if __name__ == '__main__':
    easylearn_logger()