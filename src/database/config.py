import psycopg2

def get_connection(db_config):
    config = db_config.copy()
    config["user"] = config.get("user", "postgres")
    return psycopg2.connect(**config)
