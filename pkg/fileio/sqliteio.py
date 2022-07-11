import sys
import os
import os.path
import sqlite3
import pickle
import datetime
import pandas


class SqliteDb(object):

    def __init__(self, dbfilename):
        fileexist = os.path.exists(dbfilename)
        self.con = sqlite3.connect(dbfilename)
        self.cur = self.con.cursor()
        # create table
        tbl_sql = """
        CREATE TABLE if not exists data (
            time text PRIMARY KEY,
            data blob)"""
        self.cur.execute(tbl_sql)

    def insert(self, index, data):
        s = pickle.dumps(data)
        insert_sql = """INSERT INTO data (time, data) VALUES (?, ?)"""
        self.cur.execute(insert_sql, (index, s))
        self.con.commit()

    def fetch(self, index):
        self.cur.execute(
            "SELECT data FROM data WHERE time = '{}'".format(index))
        result = self.cur.fetchone()
        data = pickle.loads(result[0])
        return data

    def drop(self):
        self.cur.execute("drop table if exists data")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.con.close()


if __name__ == '__main__':
    df = pandas.DataFrame({
        'a': list(range(10)),
        'b': list(range(10)),
        'c': list(range(10)),
    })

    dt = datetime.datetime.now()

    with SqliteDb('database.h5') as dbobj:
        print(dbobj)

        dbobj.insert(dt, df)

        data = dbobj.fetch(dt)

        print(data)

        dbobj.drop()
