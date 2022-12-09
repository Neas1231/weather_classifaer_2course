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
        "flask==2.0.0",
        "numpy>=1.17.4",
        "tensorflow>=1.15.2",
        "opencv-python-headless",
        "werkzeug==0.16.0",
        "matplotlib==3.1.1",
        "gunicorn==19.5.0",
        "Pillow>=6.0.0",
        "image",
        "scipy",
        "streamlit >= 0.63",
        "PIL",
        "st_on_hover_tabs"
    ],
)
