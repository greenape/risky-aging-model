import sqlite3
import fnmatch
import os
import argparse


def merge_db(target_db, source):
    t = sqlite3.connect(target_db)
    c = t.cursor()

    c.execute("pragma synchronous = off;")
    c.execute("pragma journal_mode=off;")

    print("Attempting to merge " + source + ".")
    query = "attach '" + source + "' as toMerge;"
    c.execute(query)
    c.execute("insert into results select * from toMerge.results;")
    c.execute("insert or ignore into parameters select * from toMerge.parameters;")
    c.execute("detach toMerge;")
    t.commit()
    c.close()
    t.close()


def merge_dbs(sources, target_db=None):
    if target_db is None:
        target_db = sources[0]
        sources = sources[1:]
    else:
        clone_empty(sources[0], target_db)
    for source in sources:
        merge_db(target_db, source)


def list_matching(directory, name):
    matching = []
    for f in os.listdir(directory):
        if fnmatch.fnmatch(f, name):
            matching.append(os.path.join(directory, f))
    return matching


def clone_empty(source, target_db):
    t = sqlite3.connect(target_db)
    c = t.cursor()

    c.execute("pragma synchronous = off;")
    c.execute("pragma journal_mode=off;")

    query = "attach '" + source + "' as toMerge;"
    c.execute(query)
    c.execute("create table if not exists results as select * from toMerge.results where 0;")
    c.execute("create table if not exists parameters as select * from toMerge.parameters where 0;")
    c.execute("detach toMerge;")
    t.commit()
    c.close()
    t.close()


def arguments():
    parser = argparse.ArgumentParser(
        description='A simple SQLite db merger.')
    parser.add_argument('-d', type=str, nargs='?',
                        help='Directory to look for DBs to merge.', default=".",
                        dest="directory")
    parser.add_argument('-f', type=str, nargs='*',
                        help='List of source filenames which may include wildcards.', default=["*.db"],
                        dest="files")

    parser.add_argument('-t', type=str, nargs='?',
                        help='Optional target DB to merge into, will be created if necessary. If ommited, uses the first matched input db.',
                        default=None,
                        dest="target")
    args = parser.parse_args()
    file_list = []
    for f in args.files:
        file_list += list_matching(args.directory, f)
    return args.target, file_list


if __name__ == "__main__":
    target, files = arguments()
    print "Merging", files
    merge_dbs(files, target)
