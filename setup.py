import setuptools

install_requires = [
    "gym",
    "pyglet",
    "box2d"
    "torch"
]


setuptools.setup(
    name='Adaptive Systems Double Deep Q Learning',
    description='Assignment for the Adaptice System coruse of the HU',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
    keyword=['Double deep q learning'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
)