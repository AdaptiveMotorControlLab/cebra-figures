import datetime

def get_years(start_year=2021):
    year = datetime.datetime.now().year
    if year > start_year:
        return f'{start_year} - {year}'
    else:
        return f'{year}'


# -- Project information -----------------------------------------------------
project = 'cebra'
copyright = f'''{get_years(2021)}, Steffen Schneider, Jin H Lee, Mackenzie Mathis.'''
author = 'Steffen Schneider, Jin H Lee, Mackenzie Mathis'
release = "0.0.1"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon',
    'sphinx_copybutton', 'nbsphinx'
]

coverage_show_missing_items = True

# Config is documented here: https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

autodoc_member_order = 'bysource'
#autodoc_typehints = "none"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'src',
    'data',
    '.*'
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# More info on theme options:
# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/configuring.html
html_theme_options = {
    'nosidebar': False,
    "icon_links": [{
        "name": "Github",
        "url": "https://github.com/stes/neural_cl",
        "icon": "fab fa-github"
    }, {
        "name": "Twitter",
        "url": "https://twitter.com/mwmathislab",
        "icon": "fab fa-twitter"
    }, {
        "name": "DockerHub",
        "url": "https://hub.docker.com/r/stffsc/cebra",
        "icon": "fab fa-docker"
    }, {
        "name": "PyPI",
        "url": "https://pypi.org/project/cebra/",
        "icon": "fab fa-python"
    }, {
        "name": "How to cite CEBRA",
        "url": "https://scholar.google.com",
        "icon": "fas fa-graduation-cap"
    }],
    "external_links": [
        #{"name": "Mathis Lab", "url": "http://www.mackenziemathislab.org/"},
        #{"name": "Bethgelab", "url": "http://bethgelab.org/"},
    ],
    "collapse_navigation": False,
    "navigation_depth": 4,
    "navbar_align": "content",
    "show_prev_next": False
}

html_logo = "_static/img/logo_large.png"

# Remove the search field for now
#html_sidebars = {'**': ['sidebar-nav-bs.html']}

# Disable links for embedded images
html_scaled_image_link = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ["css/custom.css"]


# Customizations
nbsphinx_kernel_name = 'python3'