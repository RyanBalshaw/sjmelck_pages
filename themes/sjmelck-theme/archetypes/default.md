---
title: "{{ replace .Name "-" " " | title }}"
publishdate: {{ .Date }}
author: dummy-name
description: dummy-description
draft: true
toc: true
tags: ["tag1", "tag2", "tag3"]
categories: ["category1"]
hasMermaid: false
_build:
  list: always
  publishResources: true
  render: always
---
