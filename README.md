"# FontLerp" 

Copyright (c) 2022, Roey Shapiro
All rights reserved.

This source code is licensed under the Apache 2.0-style license found in the
LICENSE.md file in the root directory of this source tree. 

This is a small project of mine that seeks to answer the question, "how does one linearly interpolate between two arbitrary vector-based fonts"?

<br>
Here's the paper on it!
https://pdfhost.io/v/7zebevzBv_FontLerp_Article_1_Export_1

<br><br>
Here are some videos of it in action!
https://www.youtube.com/playlist?list=PLRIDiXsuDNc0miu8VrzF59OjqiXGUh6ZN

<br>

Almost all of the code is mine; I used ttfQuery (ripped it apart a bit for my specific needs...) to pull font data from TTF files, and, obviously, didn't create any of the fonts themselves. They're just in there for testing purposes.

<br>

This isn't particularly user-friendly, but for those interested, it does work... mostly. The only applicable projection method is what it's currently set to: "Pillow Projection", which is set via string at the start of main.py. 
