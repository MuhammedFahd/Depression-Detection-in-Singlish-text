import psycopg2

conn = psycopg2.connect(database="dep_detector", 
                        user="fahd",
                        password="fahd123", 
                        host="35.232.162.193", port="5432")
  
cur = conn.cursor()

# if you already have any table or not id doesnt matter this 
# will create a products table for you.
cur.execute(
    '''CREATE TABLE IF NOT EXISTS products (id serial \
    PRIMARY KEY, name varchar(100), price float);''')
  
conn.commit()
  
cur.close()
conn.close()