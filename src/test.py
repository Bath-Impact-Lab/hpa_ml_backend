from src.data_processing.add_data_pipeline import add_data_pipeline

if __name__ == '__main__':
    pipeline = add_data_pipeline()
    pipeline.execute("http://images.proteinatlas.org/5583/18273_B_6_2.jpg")