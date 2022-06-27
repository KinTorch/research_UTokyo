import logging
import logging.config
from datetime import datetime
import os


class init_logger():
    def __init__(self, target: int, timenow = None):  # 0 server 1 client
        logging.config.fileConfig('logger.cfg')
        
        formatter = logging.Formatter(
            '%(asctime)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.target = target

        if target == 0:
            timenow = datetime.now()
            s_logger = logging.getLogger('server')
            self.log_path = 'logs/{:%Y-%m-%d-%H-%M-%S}/server'.format(timenow)
            os.makedirs(self.log_path, exist_ok=True)
            server_filehd = logging.FileHandler(
                os.path.join(self.log_path, 'output.log'))
            server_filehd.setFormatter(formatter)
            s_logger.addHandler(server_filehd)
            self.logger = s_logger

        elif target == 1:
            assert timenow != None
            c_logger = logging.getLogger('client')
            self.log_path = 'logs/{:%Y-%m-%d-%H-%M-%S}/clients'.format(timenow)
            os.makedirs(self.log_path, exist_ok=True)
            client_filehd = logging.FileHandler(
                os.path.join(self.log_path, 'output.log'))
            client_filehd.setFormatter(formatter)
            c_logger.addHandler(client_filehd)
            self.logger = c_logger

        else:
            raise ValueError('wrong logging target')

    def get_logger(self):
        return self.logger

    def get_path(self):
        return self.log_path

    def get_target(self):
        return self.target
