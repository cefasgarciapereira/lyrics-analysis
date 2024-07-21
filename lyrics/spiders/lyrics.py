from pathlib import Path
import sys
import re

import scrapy


class LyricsSpider(scrapy.Spider):
    name = "lyrics"
    base_url="https://www.letras.mus.br"

    def start_requests(self):
        genre = getattr(self, "genre", None)
        url = f"{self.base_url}/mais-acessadas/{genre}/"
        yield scrapy.Request(url=url, callback=self.parse_href)

    def parse_href(self, response):
        page = response.url.split("/")[-2]
        filename = f"quotes-{page}.html"
        Path(filename).write_bytes(response.body)
        self.log(f"Saved file {filename}")

        list_component = response.css('ol > li')

        for item in list_component:
            yield scrapy.Request(f"{self.base_url}{item.css('a')[0].css('a::attr(href)').extract()[0]}", callback=self.parse_lyrics)
    
    def parse_lyrics(self, response):
        title = self.trim_spaces(response.css('h1.textStyle-primary::text').get())
        author = self.trim_spaces(response.css('h2.textStyle-secondary::text').get())
        
        paragraphs = response.css('div.lyric-original > p::text').getall()
        lyric = self.trim_spaces(" ".join(paragraphs))

        yield {
            "title": title,
            "author": author,
            "lyric":lyric,
            "href": response.url
        }

    def trim_spaces(self, s):
        return re.sub(r'^\s+|\s+$', '', s)