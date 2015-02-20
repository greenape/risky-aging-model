import gzip
import sqlite3
try:
    import scoop
    scoop.worker
    single_db = False
    LOG = scoop.logger
except:
    single_db = True
    import multiprocessing
    LOG = multiprocessing.get_logger()
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

    def write_db(self, db_name, timeout=10):
        """
        Write this result set to an sqlite db.
        Creates the db if it does not exist, and if it does
        will attempt to add any missing columns.
        Will fail if the target db has columns not described in
        this resultset.
        """
        if not single_db:
            db_name = "%s_%s" % (db_name, scoop.worker[0])
        LOG.debug("Working with database %s" % db_name)

        self.make_tables(db_name, timeout)

        missing = self.check_columns(db_name, "results", self.fields, timeout)
        if len(missing) > 0:
            self.add_columns(db_name, "results", missing, timeout)
        missing = self.check_columns(db_name, "parameters", list(self.param_fields), timeout)
        if len(missing) > 0:
            self.add_columns(db_name, "parameters", missing, timeout)

        self.do_write(db_name, timeout)


    def check_columns(self, db_name, table, fields, timeout):
        """
        Return a list of columns missing from the target database table.
        """
        conn = sqlite3.connect("%s.db" % db_name, timeout=timeout)
        query = "PRAGMA table_info('%s');" % table
        result = [x[1] for x in conn.execute(query).fetchall()]
        conn.close()
        LOG.debug("DB columns are %s" % ", ".join(result))
        LOG.debug("Fields are %s" % ", ".join(fields))
        missing = filter(lambda x: x not in result, fields)
        return missing

    def add_columns(self, db_name, table, columns, timeout):
        """
        Add columns to the database table.
        """
        conn = sqlite3.connect("%s.db" % db_name, timeout=timeout)
        LOG.info("Adding columns %s to %s.%s" % (", ".join(columns), db_name, table))
        for field in columns:
            with conn:
                alter_query = 'ALTER TABLE %s ADD COLUMN %s;' % (table, field)
                conn.execute(alter_query)
        conn.close()

    def make_tables(self, db_name, timeout):
        """
        Create or ignore the required fields.
        """        
        res_fields = ",".join(self.fields)
        #print fields
        res_query = "CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, %s)" % res_fields
        
        param_fields = list(self.param_fields)
        param_fields.append("%s PRIMARY KEY" % param_fields.pop())
        param_fields = ",".join(param_fields)

        params_query = "CREATE TABLE IF NOT EXISTS parameters (%s)" % param_fields

        LOG.debug("Results table query: %s" % res_query)
        LOG.debug("Parameters table query: %s" % params_query)

        conn = sqlite3.connect("%s.db" % db_name, timeout=timeout)
        with conn:
            conn.execute(res_query)
            conn.execute(params_query)
        conn.close()

    def do_write(self, db_name, timeout):

        params = map(lambda x: tuple(map(self.type_safety, x)), self.parameters.values())
        #params = map(tuple, self.parameters.values())
        param_fields = ", ".join(self.param_fields)
        placeholders = ",".join(['?']*len(self.param_fields))
        insert_params = "INSERT OR IGNORE INTO parameters (%s) VALUES(%s)" % (param_fields, placeholders)
        LOG.debug("Params insert query: %s", insert_params)
        #print insert

        res_fields = ", ".join(self.fields)
        results = map(lambda x: tuple(map(self.type_safety, x)), self.results)
        #results = map(tuple, self.results)
        placeholders = ",".join(['?']*len(self.fields))
        insert_results = "INSERT INTO results (id, %s) VALUES(NULL, %s)" % (res_fields, placeholders)
        LOG.debug("Results insert query: %s", insert_results)
        #print insert
        conn = sqlite3.connect("%s.db" % db_name, timeout=timeout)
        with conn:
            conn.executemany(insert_params, params)
            conn.executemany(insert_results, results)
            
        conn.close()

    def type_safety(self, x):
        """
        Attempt to forcibly convert a value to a float, if that fails
        return the original value.
        This is to avoid the unpleasant scenario where values get cast to
        ints when importing into R using sqldf.
        """
        if type(x) is not str or type(x) is not unicode:
            try:
                x = float(x)
            except ValueError:
                pass
        return x




