from database import engine
from models import Base
import models

Base.metadata.create_all(bind=engine)
print("Tables created")