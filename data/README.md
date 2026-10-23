# Financial PhraseBank

Источник: [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank).
Версия репозитория: `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`.
Из исходного архива v1.0 взята часть `Sentences_AllAgree.txt`: 2264 предложения,
для каждого из которых все разметчики согласились с меткой тональности.

Авторы: Pekka Malo, Ankur Sinha, Pyry Takala, Pekka Korhonen, Jyrki Wallenius.
Публикация: *Good debt or bad debt: Detecting semantic orientations in economic
texts*, Journal of the Association for Information Science and Technology,
2014. [Статья](https://arxiv.org/abs/1307.5336).

Данные преобразованы из Latin-1 и формата `sentence@sentiment` в UTF-8 CSV
с колонками `text` и `sentiment`, без изменения меток.
Происхождение и контрольные суммы: `raw/financial_news.source.json`.
Очистка и удаление дубликатов выполняются при обучении; исходный CSV сохранён.

Лицензия данных — [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
Условия источника сохранены в [LICENSE.txt](LICENSE.txt).
К включённым данным и производным материалам на их основе, в том числе
обученной модели и примерам из корпуса в отчётах, применяется эта лицензия.
