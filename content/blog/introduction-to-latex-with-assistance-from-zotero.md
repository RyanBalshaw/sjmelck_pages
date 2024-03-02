---
title: "Introduction to Latex With Assistance From Zotero"
publishdate: 2024-02-24T16:37:48+02:00
author: Justin Smith
description: A blog post on using Zotero and research rabbit.
draft: false
toc: true
tags: ["LaTeX", "Report Writing", "Research Rabbit", "Researching More Efficiently", "Zotero"]
categories: ["category1"]
_build:
  list: always
  publishResources: true
  render: always
---
Good day 👋

🧠 This blog aims to provide an introduction towards using LaTeX for reports in conjunction with Zotero as a reference manager and Research rabbit to aid “fast-tracking” research.

## Outcomes:

- Understand how Zotero can be used as a powerful research tool 💪.
- Know how to configure Zotero to use a `.bib` file containing stored references that can be used when typesetting with LaTeX 💁.
- Be aware of the basics of other artificial intelligence (AI) research tools such as Research Rabbit 🐇.

# Zotero as a research tool with LaTeX

[Zotero](https://www.zotero.org/) is not just another tool; it's a research companion, a digital librarian, and an organizational workspace all in one environment. This section presents how Zotero can improve ones engagement with literature in conjunction with making the connection between reading literature and referencing using LaTeX as a typesetting tool.

## What is Zotero

Zotero is an open-source reference management software designed to help researchers and scholars organize and cite their sources. It was developed by the Roy Rosenzweig Center for History and New Media at George Mason University and was first released in 2006. Zotero provides a convenient way to collect, store, and organize bibliographic information, including journal articles, books, websites, and other types of research materials.

Key features of Zotero include:

1. **Web Browser Integration**: Zotero integrates with web browsers like Firefox, Chrome, and Safari, allowing users to capture citation information and full-text content directly from webpages.
2. **Citation Management**: Zotero allows users to create and manage citations and bibliographies in various citation styles, including IEEE, APA, MLA, Chicago, and more.
3. **Organization**: Users can organize their research materials into collections, tags, and subcollections, making it easy to categorize and find information.
4. **Syncing and Backup**: Zotero provides cloud-based syncing and backup options, enabling users to access their library and data from different devices.
5. **Collaboration**: Users can collaborate on research projects by sharing collections with others and working on them simultaneously.
6. **PDF Annotation**: Zotero allows users to annotate PDF documents within the application, making it convenient to take notes and highlight important information.

Overall, Zotero is a powerful tool for researchers, students, and academics who need to manage and cite their sources effectively, improving the efficiency of their research workflows. However, please note that software features and updates might have occurred after my last update, so I recommend checking the official Zotero website or other reliable sources for the most current information about the software.

## Zotero vs Mendeley

Zotero is analogous to Mendeley from a reference manager perspective. The information presented for setting up a `.bib` file can also be done with Mendeley. The main difference is the storage limitations between Zotero and Mendeley, and of course Zotero is open source.

The limited cloud storage offered by Zotero means that you might be required to buy additional storage. However I feel the benefits of having a dedicated mobile app, for Zotero, along with the software being open source outweighs this price. Since Zotero is open source there are a lot of other software using Zotero as a reference manager in all interesting ways, one example being Research Rabbit.

However if this is of no interest to you Mendeley might be a better option, especially from a cost perspective. Whichever option you go for - the most important thing is to ensure you stick with it and use the reference manager for all editing. Albeit PDF annotations and reading articles in depth.

## How to use Zotero effectively for research in conjunction with LaTeX

As previously alluded to, Zotero is a reference manager which can function seemlessly with typesetting in LaTeX. The primary benefit of this approach is that it allows you to add references in an analogous manner across all the reports you build throughout your research. It is a simple and easy way to also avoid generating unique `.bib` files for each report you write by referencing a single common `.bib` file stored in a fixed location.

### Making a `.bib` file which is synchronized and updated appropriately

My personal preference is to have Zotero automatically update a `.bib` reference file (containing **all** my citations across all my research) which is in a fixed location (on my computer). Consequently, every time I make a new report in LaTeX, I just copy + paste three lines of code to add citations throughout the report. I found this particularly useful during my post-graduate (Honours and Masters) studies when one has to compile many reports across the board of modules.

To do this:

- Navigate to Zotero’s home page and click `file/Export Library…`
- Use the *Format* `Better BibLaTex`. The preambles use this.
- Thereafter, export the library to your preferred location with the `Keep Updated` **option *enabled*. This ensures the `.bib` file is synchronized every time you add a new reference to Zotero.
- Then choose where you prefer to have your `.bib` file stored. Note this location as its crucial for defining your `.bib` file in your LaTeX document.


> Note that you will be required to install the Better BibTeX for Zotero plugin to do this. Please see <a href="https://retorque.re/zotero-better-bibtex/">this link on how to install Better BibTeX</a> as well as <a href="https://retorque.re/zotero-better-bibtex/">this link on how to install Better BibTeX</a>.

Although I strongly recommend formatting each citation properly in Zotero - translating to a well formatted `.bib` file - it is possible to export a collection (subfolder) in Zotero. This could be done for those wanting to group all references relating to a single article they are wishing to publish and in the case where there is no workaround for having to edit directly inside the `.bib` file.

> 👍 Unfortunately setting up (and exporting) multiple libraries are not yet available and hopefully this will change in the near future as discussed [here](https://forums.zotero.org/discussion/104385/multiple-local-libraries#:~:text=Currently%20there%20is%20no%20way,stored%20in%20the%20same%20folder).

Please also see this guide for more information on exporting specific references
[here](https://guides.library.ucsc.edu/c.php?g=240807&p=9193492#:~:text=Exporting%20Zotero%20to%20BibTeX,%2Dclicks%20and%20shift%2Dclicks.&text=Select%20the%20BibTeX%20format%20and%20click%20OK).

### Preambles in LaTeX

Key preambles into latex:

The following preambles are required for LaTeX to use your new `.bib` file for referencing.

```latex
% Referencing
\usepackage[backend=biber,style=ieee]{biblatex} % Configures the bibliography and citations of your latex document.
\addbibresource{C:/.../Zotero/MyZoteroLibrary.bib} % Specifies the path to the .bib file
```

The rationale for using biblatex (over Biber or perhaps natbib) is because this works with both bibtex and biblatex which seemed to be the best format for the Zotero plugin that is available and is also the most customisable format supporting the most datafields required by some citation schemes. More information with a detailed summary on the different formats is described on this [StackOverflow](https://tex.stackexchange.com/questions/25701/bibtex-vs-biber-and-biblatex-vs-natbib) post.

> 😇 Please note that the directory should correspond to the specific `.bib` file you are using for your article/report/book.

Once these preambles have been added, the `\cite{CitationKey}` command can be used to add citations (with citation key generated by Zotero) into the document.

### Other important settings

Depending on the TeX editor of choice (I prefer [TeXstudio](https://www.texstudio.org/)), you might have to tinker with the applications `commands` and ensure that the LaTeX document is compiled and turned into a PDF twice to allow for the references to appear in the PDF appropriately.

> :anatomical_heart: Note to use TeXstudio, you will also need to download [MiKTeX](https://miktex.org/download) to handle all the LaTeX packages you use.

Some settings in TeXstudio’s `command` panel might be required to be updated. Specifically, the PdfLaTeX is required to be set to:
`pdflatex -synctex=1 -interaction=nonstopmode --shell-escape %.tex`
This ensures that you can add citations with the method described in this blog.

![TeXstudio command window](TeXstudio_command_window.png)*Figure showing TeXstudio `command` window, some changes might need to be made here if you are currently running default settings*

> 🧑‍🔧 This [StackOverflow](https://tex.stackexchange.com/questions/135102/biblatex-doesnt-show-bibliography-when-compiling#:~:text=In%20TeXstudio%2C%20you%20can%20force,repeatedly%20every%20time%20you%20compile) post provides some help if BibLaTeX doesn't show the bibliography when compiling in TeXstudio.


You might also be required to change the `Build` configuration in your TeX editor as shown below:

![`Build` *Window in TeXstudio which might be required to be updated.*](TeXstudio_build_window.png)*Figure showing TeXstudio `build` window, some changes might need to be made here if you are currently running default settings*

> 🫸 Note that this build window might also be required to be updated depending on which packages you are using. For example, for the glossaries package certain settings might be required to be adjusted (see this [StackOverflow](https://tex.stackexchange.com/questions/156270/has-anyone-managed-to-use-glossaries-with-texstudio-on-windows) post for more information).

## Research admin in Zotero

To be as efficient as possible, admin should be performed on a continual basis. Doing a few simple tasks each time you add an article to your Zotero workspace will really assist with navigating through your research library.

### Using Zotero as a PDF reader

As previously alluded to, the Zotero workspace should serve as the primary space for all your reference and research management needs. To be as efficient as possible, the article’s PDF should rather be added to Zotero and annotations/highlights should be performed on the PDF stored in the Zotero cloud. Future issues can be prevented by ensuring that you keep to using the on-board features provided by Zotero directly.

### Tagging articles

Tagging articles and research papers ensures that you can easily filter research outputs according to specific keywords. The objective is to make the tags as simple as possible and have a consistent nomenclature. This also ensures that the tags are as robust as possible to being used as a filter/sorting mechanism.

Some tagging strategies follow keeping the tags as concise as possible and not too general. For example “vibration” is a poor tag if your research is focused on vibrations and vibration based condition monitoring. A more appropriate tag might be “Vibration Envelope Analysis” or “Bearing Fault Diagnostics”. This way the keyword “Vibration” is still in the tag (aiding filtering/searching) but the tag is robust enough towards more specific searches.

Furthermore, avoid adding tags containing keywords in another descriptor of the article. For example do not tag using words/phrases which appear in the title. The aim is to be able to search across all fields describing every article and find the article after typing as few words as possible (obviously).

# Research Rabbit
[![Research rabbit clickable link](research_rabbit_logo.png)](https://www.youtube.com/watch?v=W1W51rYJA3I&ab_channel=ResearchRabbit)

The following section describes how to use [Research Rabbit](https://www.researchrabbit.ai/) as a tool for discovering new academic articles in conjunction with Zotero as a reference manager.

Research Rabbit is a reasonably new AI tool which allows you to quickly find citations relating to an article you’re currently interested in. RR then generates a mindmap like structure of all the academic articles relating to the current paper of interest. What makes this tool particularly useful is that it can synchronize seamlessly with your Zotero references and even Zotero’s collections. This means that if the article is added as a new citation in Zotero (provided the collection is being synchronized with RR) the citation will also appear in RR and you can then use it as a benchmark paper to discover new research outputs.

The video below provides more assistance on integrating Zotero with Research Rabbit.

[![Image alt text](integrating_zotero_with_research_rabbit.jpg)](https://www.youtube.com/watch?v=W1W51rYJA3I&ab_channel=ResearchRabbit)

---
As always, thanks for reading! 👨‍💻

Please feel free to contact me ([Justin Smith](mailto:66smithjustin@gmail.com?subject=Sjmelck blog post)) for any suggestions or comments!
