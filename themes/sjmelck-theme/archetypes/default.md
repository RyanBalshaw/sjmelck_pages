---
title: "{{ replace .Name "-" " " | title }}"
publishdate: {{ .Date }}
author: dummy-name
description: dummy-description
draft: true
toc: true
_build:
  list: always
  publishResources: true
  render: always
---
