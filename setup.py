from setuptools import setup, find_packages

setup(name='mltools',
      version='0.2',
      description='A library for data processing',
      url='https://github.com/p2327/mltools',
      author='Pietro Pravettoni',
      author_email='@example.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'pandas',
            'pytest', 
            'scikit-learn', 
            'sklearn_pandas',
            'typing'
            ],
      include_package_data=True,
      python_requires='>=3.6.1',
      zip_safe=False)


    
