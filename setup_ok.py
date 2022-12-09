import setuptools

setuptools.setup(
    name="streamlit-weather",
    version="0.0.2",
    author="",
    author_email="",
    description="",
    long_description="",
    long_description_content_type="text/plain",
    url="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[],
    python_requires=">=3.6",
    install_requires=[
      tensorflow==2.4.0
      numpy==1.19.4
      streamlit==0.73.1
      pandas==1.1.4
      Pillow==8.2.0
    ],
)
