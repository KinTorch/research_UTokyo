import logging
import logging.config
from datetime import datetime
import os

class init_logger():
    def __init__(self):
        logging.config.fileConfig('logger.cfg')
        s_logger = logging.getLogger('server')
        c_logger = logging.getLogger('client') #bug: lose handler 
        timenow = datetime.now()
        self.server_path = 'logs/{:%Y-%m-%d-%H-%M-%S}/server'.format(timenow)
        self.clients_path = 'logs/{:%Y-%m-%d-%H-%M-%S}/clients'.format(timenow)
        os.makedirs(self.server_path)
        os.makedirs(self.clients_path)
        server_filehd = logging.FileHandler(os.path.join(self.server_path,'output.log'))
        client_filehd = logging.FileHandler(os.path.join(self.clients_path,'output.log'))
        formatter = logging.Formatter('%(asctime)s|%(message)s',datefmt='%Y-%m-%d %H:%M:%S')
        server_filehd.setFormatter(formatter)
        client_filehd.setFormatter(formatter)
        s_logger.addHandler(server_filehd)
        c_logger.addHandler(client_filehd)
        self.s_logger = s_logger
        self.c_logger = c_logger
    
    def get_server_logger(self):
        return self.s_logger
    
    def get_clients_logger(self):
        return self.c_logger

    def get_server_path(self):
        return self.server_path

