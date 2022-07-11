import os
import os.path
import logging

def init_logging(logfn):
    if os.path.exists(logfn):
        pass
    else:
        parent = os.path.dirname(logfn)
        if len(parent) > 0:
            os.makedirs(parent, exist_ok=True)
    logging.basicConfig(
        filename=logfn,
        filemode='w+',
        format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
    )


def log(*ls):
    line = [str(i) for i in ls]
    line = ' '.join(line)
    logging.info(line)


def log_str(*ls):
    line = [str(i) for i in ls]
    line = ' '.join(line)
    logging.info(line)
    print(line)


def test_log():
    import time
    logfn = 'log/log.log'
    init_logging(logfn)
    log_str('this', 'is', 'a', 'test')
    time.sleep(1)
    #assert os.path.exists(logfn) == True, "success"
