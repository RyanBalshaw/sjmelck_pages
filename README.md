<!-- Begin section: Overview -->

[//]: # (https://raw.githubusercontent.com/jehna/readme-best-practices/master/sample-logo.png&#41;)

[//]: # (![Logo of the project]&#40;https://raw.githubusercontent.com/RyanBalshaw/sjmelck_pages/main/robot_logo.svg&#41;)

# Sjmelck
> Sjmelck just means engineers love computers, k?

<br/>

View the page [here](https://ryanbalshaw.github.io/sjmelck_pages/)

![GitHub last commit](https://img.shields.io/github/last-commit/RyanBalshaw/sjmelck_pages?color=important)
![GitHub contributors](https://img.shields.io/github/contributors/RyanBalshaw/sjmelck_pages?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/RyanBalshaw/sjmelck_pages?color=critical&style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/RyanBalshaw/sjmelck_pages?color=blueviolet&style=flat-square)
![GitHub license](https://img.shields.io/github/license/RyanBalshaw/sjmelck_pages?style=flat-square)


The purpose of Sjmelck is to provide an awesome blog for content related to engineering, computing, and everything in-between.

## Installing / Getting started

A quick introduction of the minimal setup you need to get a blog post up and running. As a first step, please make sure the following packages are installed:
1. [Hugo](https://gohugo.io/)
2. [git](https://git-scm.com/)

Once these have been installed, you are welcome to run the following:
```shell
git clone https://github.com/RyanBalshaw/sjmelck_pages.git
cd sjmelck_pages
hugo server # Creates a local version
```

This will clone the repo and open up a local version of the website. Press `Ctrl + C` to stop the local server.

## Creating a blog post

If you wish to create a blog post, it is as simple as navigating to the base directory and running:

```shell
hugo new blog/a-new-blog-post.md
```

This will create a new file in the `contents/blog` directory. You can then open the file with any editor of your choosing.

```
---
title: "A Trial Blog Post"
publishdate: 2023-04-20T16:19:31+02:00
author: dummy-name
description: dummy-description
draft: true
toc: true
tags: ["tag1", "tag2", "tag3"]
categories: ["category1"]
_build:
  list: always
  publishResources: true
  render: always
---
```

This information will be at the start of the file. Please change the information as necessary. Importantly, if you wish to push your blog post to the main branch you will need to change `draft: false` to `draft: true`. You can then add [markdown](https://commonmark.org/help/) to the post.

Save the file, then start a local server to view the changes you made.
```shell
hugo server --buildDrafts
hugo server -D
```

View the blog post and make changes as necessary.

## Adding equations in markdown

Adding equations is simple, and [Mathjax]() is used to support any equations. Inline equations can be created using
```markdown
\\( ... \\)
```
and equations can be created using
```markdown
\[ ... \]
```
or
```markdown
$$ ... $$
```

More information can be found [here](https://docs.mathjax.org/en/latest/input/tex/delimiters.html)

## Contributing

Johann testing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.

[//]: # (## Links)

[//]: # ()
[//]: # (Even though this information can be found inside the project on machine-readable)

[//]: # (format like in a .json file, it's good to include a summary of most useful)

[//]: # (links to humans using your project. You can include links like:)

[//]: # ()
[//]: # (- Project homepage: https://your.github.com/awesome-project/)

[//]: # (- Repository: https://github.com/your/awesome-project/)

[//]: # (- Issue tracker: https://github.com/your/awesome-project/issues)

[//]: # (  - In case of sensitive bugs like security vulnerabilities, please contact)

[//]: # (    my@email.com directly instead of using issue tracker. We value your effort)

[//]: # (    to improve the security and privacy of this project!)

[//]: # (- Related projects:)

[//]: # (  - Your other project: https://github.com/your/other-project/)

[//]: # (  - Someone else's project: https://github.com/someones/awesome-project/)


## Licensing

The code in this project is licensed under MIT license.

<!-- End section: Overview -->
