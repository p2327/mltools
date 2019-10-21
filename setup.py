from setuptools import setup

setup(name='mltools',
      version='0.1',
      description='A library for data processing',
      url='',
      author='Pietro Pravettoni',
      author_email='@example.com',
      license='MIT',
      packages=['mltools'],
      install_requires=['pandas', 'typing',
                        'sklearn', 'sklearn_pandas'],
      include_package_data=True,
      zip_safe=False)


    
