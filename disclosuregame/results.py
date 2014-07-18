import gzip
import sqlite3
try:
    import scoop
    scoop.worker
    single_db = False
except:
    single_db = True
    pass

class Result(object):
    def __init__(self, fields, parameters, results):
        fields.append("hash")
        self.fields = fields
        self.param_fields = parameters.keys()
        self.param_fields.append("hash")
        param_hash = "h%d" % hash(tuple(parameters.values()))
        self.parameters = {param_hash:parameters.values() + [param_hash]}
        for result in results:
            result.append(param_hash)
        self.results = results

    def add_results(self, results):
        self.parameters.update(results.parameters)
        self.results += results.results
        return self

    def write(self, file_name, sep=","):
        """
        Write a results to a (csv) file.
        """
        if not single_db:
            file_name = "%s_%s" % (scoop.worker[0], file_name)
        result = [sep.join(self.fields)]
        result += map(lambda l: sep.join(map(str, l)), self.results)
        file = gzip.open(file_name, "w")
        file.write("\n".join(result))
        file.close()
    
    def write_params(self, file_name, sep=","):
        if not single_db:
            file_name = "%s_%s" % (scoop.worker[0], file_name)
        result = [sep.join(self.param_fields)]
        result += map(lambda l: sep.join(map(str, l)), self.parameters.values())
        file = gzip.open(file_name, "w")
        file.write("\n".join(result))
        file.close()

    def write_db(self, db_name):
        """
        Write this result set to an sqlite db.
        Creates the db if it does not exist, and if it does
        will attempt to add any missing columns.
        Will fail if the target db has columns not described in
        this resultset.
        """
        if not single_db:
            db_name = "%s_%s" % (db_name, scoop.worker[0])

        conn = sqlite3.connect("%s.db" % db_name)
        
        fields = ",".join(self.fields)
        #print fields
        conn.execute("CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, %s)" % fields)
        
        fields = list(self.param_fields)
        fields.append("%s PRIMARY KEY" % fields.pop())
        fields = ",".join(fields)
        conn.execute("CREATE TABLE IF NOT EXISTS parameters (%s)" % fields)
        conn.close()

        missing = self.check_columns(db_name, "results", self.fields)
        if len(missing) > 0:
            self.add_columns(db_name, "results", missing)
        missing = self.check_columns(db_name, "parameters", list(self.param_fields))
        if len(missing) > 0:
            self.add_columns(db_name, "parameters", missing)

        conn = sqlite3.connect("%s.db" % db_name)

        params = map(tuple, self.parameters.values())
        placeholders = ",".join(['?']*len(self.param_fields))
        insert = "INSERT OR IGNORE INTO parameters VALUES(%s)" % placeholders
        #print insert
        conn.executemany(insert, params)

        results = map(tuple, self.results)
        placeholders = ",".join(['?']*len(self.fields))
        insert = "INSERT INTO results VALUES(NULL, %s)" % placeholders
        #print insert
        conn.executemany(insert, results)
        conn.commit()
        conn.close()

    def check_columns(self, db_name, table, fields):
        """
        Return a list of columns missing from the target database table.
        """
        conn = sqlite3.connect("%s.db" % db_name)
        query = "PRAGMA table_info('%s');" % table
        result = [x[1] for x in conn.execute(query).fetchall()]
        conn.close()
        missing = filter(lambda x: x not in result, fields)
        return missing

    def add_columns(self, db_name, table, columns):
        """
        Add columns to the database table.
        """
        conn = sqlite3.connect("%s.db" % db_name)
        c = conn.cursor()
        for field in columns:
            try:
                alter_query = 'ALTER TABLE %s ADD COLUMN %s;' % (table, field)
                c.execute(alter_query)
            except:
                pass # handle the error
        c.commit()
        c.close()


