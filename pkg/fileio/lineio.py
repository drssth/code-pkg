import os
import os.path
import logging


class LineReader(object):
    def __init__(self, fn):
        self.f = open(fn, 'r')


    def get_lines(self):
        for line in self.f:
            yield line


    def __del__(self):
        self.f.close()


class LineWriter(object):
    def __init__(self, fn, overwrite=False):
        self.fn = fn
        self.parent = os.path.dirname(self.fn)
        self.create_file = True

        if os.path.exist(self.fn):
            if overwrite:
                self.create_file = True
            else:
                self.create_file = False
        else:
            self.create_file = True

        if self.create_file:
            logging.info('create new file', self.fn)
            os.makedirs(self.parent, exist_ok=True)
            with open(self.fn, 'w+') as f:
                f.write('### start ###')
                f.write('\n')
        else:
            logging.info('resume on existing file', self.fn)
            with open(self.fn, 'a+') as f:
                f.write('### resume ###')
                f.write('\n')

    def write(self, *ls):
        line = [str(i) for i in ls]
        line = ' '.join(line)
        with open(self.fn, 'a+') as f:
            f.write(line)
            f.write('\n')
