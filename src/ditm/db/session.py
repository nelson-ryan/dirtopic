from functools import cache
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker


@cache
def get_engine(db_url):
    return create_engine(db_url)


@cache
def get_sessionmaker(db_url):
    engine = get_engine(db_url)
    return sessionmaker(engine)
