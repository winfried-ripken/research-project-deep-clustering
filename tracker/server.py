import json
import time
import sqlite3
import psutil
import GPUtil

from threading import Thread
from flask import request, redirect
from flask import g
from flask import Flask

app = Flask(__name__)

DATABASE = 'workers.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route("/")
def hello():
    result = "<!DOCTYPE html><html><head><title>Dashboard</title><meta content='2; url=http://127.0.0.1:5000/' http-equiv='refresh'>"\
           + "<style>th, td { padding: 15px; }</style></head><body>"

    result += str(time.asctime(time.localtime(time.time()))) + "</br>" + "</br>"

    cmd = "select * from worker_conf Where Active=1 order by Id;"

    conn = get_db()
    cur = conn.cursor()
    cur.execute(cmd)

    result += "<h2>Running instances:</h2><table>"

    for item in cur.fetchall():
        result += "<tr>"
        result += "<td>"
        result += "<a href='/details?id=" + str(item[0]) + "'>" + item[2] + "</a>"
        result += "</td>"

        result += "<td>"
        result += "<progress value='"+str(item[1])+"' max='100'></progress>"
        result += "</td>"

        result += "<td>"

        cmd2 = "select * from worker where Id=? order by Time DESC LIMIT(1);"
        cur2 = conn.cursor()
        cur2.execute(cmd2, (item[0], ))

        for item2 in cur2.fetchall():
            result += "Status @" + str(time.asctime(time.localtime(item2[2]))) + "</br>" + str(item2[1])

        result += "</td>"

        result += "</tr>"
    result += "</table>"

    result += "<h2>System stats:</h2>"

    result += "CPU usage: " + str(psutil.cpu_percent()) + "%</br>"
    result += "Memory: " + str(psutil.virtual_memory()[2]) + "%</br>"
    
    gd = read_gpu()
    result += "GPU usage: " + str(gd[0]) + "</br>"
    result += "GPU memory: " + str(gd[1]) + "</br>"

    result += "<h2>Finished instances:</h2><table>"

    cmd = "select * from worker_conf Where Active=0 order by Id;"
    cur = conn.cursor()
    cur.execute(cmd)

    for item in cur.fetchall():
        result += "<tr>"
        result += "<td>"
        result += "<a href='/details?id=" + str(item[0]) + "'>" + item[2] + "</a>"
        result += "</td>"
    result += "</table>"

    result += "</body>"
    return result

@app.route("/delete")
def delete():
    id = request.args.get('id')
    cmd = "DELETE from worker_conf where Id = ?;"
    cmd2 = "DELETE from worker where Id = ?;"

    conn = get_db()
    conn.execute(cmd, (id, ))
    conn.execute(cmd2, (id, ))
    conn.commit()

    return redirect("http://127.0.0.1:5000/", code=302)

@app.route("/create_db")
def db():
    cmd = "CREATE TABLE worker (Id Varchar, Status Varchar, Time real);"
    cmd2 = "CREATE TABLE worker_conf (Id Varchar, Progress Int, Details Varchar, Active Boolean);"

    conn = get_db()
    conn.execute(cmd)
    conn.execute(cmd2)
    conn.commit()

    return "Successful"

@app.route('/details', methods=['GET'])
def parse_details():
    id = request.args.get('id')

    result = "<!DOCTYPE html><html><head><title>Details</title>" \
             + "</head><body>"

    conn = get_db()
    cmd = "select * from worker_conf Where Id = ?;"
    cur = conn.cursor()
    cur.execute(cmd, (id,))

    for item in cur.fetchall():
        result += str(item[2]) + "</br>"

    result += "<a href='/delete?id=" + str(id) + "'>DELETE</a>"
    result += "</br></br>"

    cmd = "select * from worker where Id = ? order by Time;"
    cur = conn.cursor()
    cur.execute(cmd, (id,))

    for item in cur.fetchall():
        result += str(item[1]) + "</br>"

    result += "</body>"
    return result

@app.route('/register', methods=['GET', 'POST'])
def parse_register():
    dataDict = json.loads(request.data)

    cmd = "Insert into worker_conf (Id, Progress, Details, Active) values (?,0,?,1)"
    conn = get_db()
    conn.execute(cmd, (dataDict["Id"], dataDict["Details"]))
    conn.commit()

    return "OK"

@app.route('/deregister', methods=['GET', 'POST'])
def parse_deregister():
    dataDict = json.loads(request.data)

    cmd = "Update worker_conf set Active=0 where Id=?"
    conn = get_db()
    conn.execute(cmd, (dataDict["Id"], ))
    conn.commit()

    return "OK"

@app.route('/status', methods=['GET', 'POST'])
def parse_request():
    dataDict = json.loads(request.data)

    cmd = "Insert into worker (Id, Status, Time) values (?,?,strftime('%s','now'))"
    conn = get_db()
    conn.execute(cmd, (dataDict["Id"], dataDict["Status"]))

    cmd = "Update worker_conf Set Progress=? Where Id=?"
    conn.execute(cmd, (dataDict["Progress"], dataDict["Id"]))
    conn.commit()

    return "OK"

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.load = "detecting.."
        self.mem = "detecting.."
        self.start()

    def run(self):
        try:
            while not self.stopped:
                gpu = GPUtil.getGPUs()[0]
                self.load = str(round(gpu.load*100,2)) + "%"
                self.mem = str(round(gpu.memoryUtil*100,2)) + "%"
                time.sleep(self.delay)

        except:
            print(self.load)
            self.load = "no gpu"
            self.mem = "no gpu"

    def stop(self):
        self.stopped = True

def read_gpu():
    if not hasattr(read_gpu, "_monitor"):
        read_gpu._monitor = Monitor(5)

    return (read_gpu._monitor.load, read_gpu._monitor.mem)