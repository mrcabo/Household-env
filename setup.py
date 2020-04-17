from setuptools import setup

setup(name='gym_foo',
      version='0.0.1',
      install_requires=['gym', 'numpy'],  # And any other dependencies foo needs
      package_data={
          # If any package contains *.txt or *.rst files, include them:
          "": ["*.txt", "*.json"]
      }
      )
