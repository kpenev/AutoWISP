<a name="top"></a>
[![AutoWISP](https://raw.githubusercontent.com/kpenev/AutoWISP/master/.github/images/AutoWISP.png)](https://github.com/kpenev/AutoWISP)
[![language](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](https://www.python.org/about/)
[![OS](https://img.shields.io/badge/OS-linux%2C%20windows%2C%20macOS-0078D4)](https://github.com/kpenev/AutoWISP/tree/master/documentation)
[![CPU](https://img.shields.io/badge/CPU-x86%2C%20x64%2C%20ARM%2C%20ARM64-FF8C00)](https://github.com/kpenev/AutoWISP/tree/master/documentation)
[![GitHub release](https://img.shields.io/github/v/release/kpenev/AutoWISP)](#)
[![GitHub release date](https://img.shields.io/github/release-date/kpenev/AutoWISP)](#)
[![GitHub last commit](https://img.shields.io/github/last-commit/kpenev/AutoWISP)](#)
[![getting started](https://img.shields.io/badge/getting_started-guide-1D76DB)](https://github.com/kpenev/AutoWISP/blob/master/autowisp/tests/test_data/test_data.ipynb)
[![Free](https://img.shields.io/badge/free_for_non_commercial_use-brightgreen)](#-license)

⭐ Star us on GitHub — it motivates us a lot!

## Table of Contents
- [About](#-about)
- [How to Install](#-how-to-install)
- [Documentation](#-documentation)
- [Feedback and Contributions](#-feedback-and-contributions)
- [License](#-license)
- [Contacts](#%EF%B8%8F-contacts)

## 🚀 About

**AutoWISP** is a Python package designed to allow users to automatically create and de-trend light curves from their astronomical data. It adheres to high standards of flexibility, reusability, and reliability, utilizing well-known software design patterns, including modular and hexagonal architectures. These patterns ensure the following benefits:

- **Modularity**: Different parts of the package can function independently, enhancing the package's modularity and allowing for easier maintenance and updates.
- **Testability**: Improved separation of concerns makes the code more testable.
- **Maintainability**: Clear structure and separation facilitate better management of the codebase.

## 📝 How to Install

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install AutoWISP.

    pip install autowisp

## 📚 Documentation

### Getting Started
For a hands-on example, you can explore our Jupyter Notebook which processes a sample dataset from start to finish.

This interactive notebook provides a practical demonstration of the AutoWISP pipeline in action.

Explore the [View the interactive Processing Example Documentation](https://github.com/kpenev/AutoWISP/blob/master/autowisp/tests/test_data/test_data.ipynb) or [View the interactive Processing Example on nbviewer](https://nbviewer.org/github/kpenev/AutoWISP/blob/main/autowisp/tests/test_data/test_data.ipynb).

In this guide, you will go step by step in the pipeline, producing the corresponding files needed for each step and ultimately creating light curves using a test dataset we provide.

To better understand the AutoWISP pipeline, we recommend visiting our [Documentation](https://github.com/kpenev/AutoWISP/tree/master/documentation) site.
There, you will find useful information about the individual steps, database, and/or browser-user-interface (currently under development).

> [!]
> All the following images were created using tools in this repository.

### Example Light Curves

![WASP-33 b with TESS](https://raw.githubusercontent.com/kpenev/AutoWISP/master/.github/images/wasp33_with_tess.png)

This is the resulting phase-folded lightcurve for WASP-33 b exoplanet transit, observed by Project PANOPTES (blue points and circles), TESS (red points), and theoretical light curve based on best known system parameters (green curve). The raw PANOPTES-DSLR measurements, originating from the 4 color channels of 4 cameras in Hawaii (Mauna Loa observatory) and California (Mt Wilson) are shown as blue points. The blue points are binned in time to create the blue circles and corresponding error bars. Note that the scatter in TESS points is not instrumental, but rather it is intrinsic variability in the host star, which is a member of the delta-Scuti class of variable stars.

### Example Photometric Precision 
![MAD Plot](https://raw.githubusercontent.com/kpenev/AutoWISP/master/.github/images/mad_plot.png)

The scatter (median absolute deviation from the median) of the individual channel lightcurves of PANOPTES observations of a $10\times15$ degree field centered on FU Orionis, with each of their corresponding image colors (RGGB). We see that AutoWISP enables a few parts per thousand photometric precision per exposure even from images with Bayer masks, significantly outperforming prior efforts. Even individual color channels result in better than 1% photometry per 2 min exposure.

## 🤝 Feedback and Contributions

We've made every effort to implement all the main aspects of AutoWISP in the best possible way. However, the development journey doesn't end here, and your input is crucial for our continuous integration and development.

> [!IMPORTANT]
> Whether you have feedback on features, have encountered any bugs, or have suggestions for enhancements, we're eager to hear from you. Your insights help us make the AutoWISP package more robust and user-friendly.

Please feel free to contribute by [submitting an issue](https://github.com/kpenev/AutoWISP/issues) or [joining the discussions](https://github.com/kpenev/AutoWISP/discussions). Each contribution helps us grow and improve.

We appreciate your support and look forward to making our pipeline even better with your help!

## 📃 License

This package is distributed under the MIT License. You can review the full license agreement at the following link: [MIT](https://github.com/kpenev/AutoWISP/blob/master/LICENSE).

This package is available for free!

## 🗨️ Contacts

For more details about our usages, services, or any general information regarding the AutoWISP pipeline, feel free to reach out to us. We are here to provide support and answer any questions you may have. Below are the best ways to contact our team:

- **Email**: Send us your inquiries or support requests at [support_autowisp@gmail.com](mailto:support_autowisp@gmail.com).

We look forward to assisting you and ensuring your experience with AutoWISP is successful and enjoyable!

[Back to top](#top)
