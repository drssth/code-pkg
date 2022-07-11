from tinydb import TinyDB, Query


class TinyDbInterface(object):
    """
    key
    status True/False
    """
    def __init__(self, path):
        self.path = path
        self.db = TinyDB(self.path)
        self.q = Query()


    def add(self, item, check=True):
        tag = 0
        if check:
            if 'key' in item:
                k = item['key']
                r = self.find(k)
                if r is None:
                    self.db.insert(item)
                    tag = True
                else:
                    tag = False
                    print('key already existed')
            else:
                tag = -1
                print('no key in item', item)
        else:
            self.db.insert(item)
        return tag


    def add_many(self, items):
        self.db.insert_multiple(items)


    def find(self, key):
        item = self.db.search(self.q.key == key)
        if len(item) > 0:
            return item[0]
        return None


    def get_status(self, key):
        r = self.find(key)
        if r is None:
            return None
        if 'status' in r:
            return r['status']
        return None


if __name__ == '__main__':
    import json

    db = TinyDbInterface('status.json')

    with open('nocontentsites.json', 'r') as f:
        ls = json.load(f)

        for k in ls[96655:]:
            db.add({'key':k,'status':False})

