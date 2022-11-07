from nltk.tokenize import word_tokenize


def block_repeat_ngrams(text, n: int = 5):
    words = word_tokenize(text)
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    tracker = {}
    tobe_removed_ngram_idxs = []
    tobe_removed_ngrams = []
    for idx, ngram in enumerate(ngrams):
        if ngram not in tracker:
            tracker[ngram] = True
        else:
            tobe_removed_ngram_idxs.append(idx)
            tobe_removed_ngrams.append(ngram)

    tobe_removed_intervals = [(idx, idx + n) for idx in tobe_removed_ngram_idxs]
    tobe_removed_intervals_merged = []
    for interval in tobe_removed_intervals:
        if len(tobe_removed_intervals_merged) == 0:
            tobe_removed_intervals_merged.append(interval)
        else:
            last_interval = tobe_removed_intervals_merged[-1]
            if last_interval[1] >= interval[0]:
                tobe_removed_intervals_merged[-1] = (last_interval[0], interval[1])
            else:
                tobe_removed_intervals_merged.append(interval)
    
    tobe_removed_word_idxs = []
    for interval in tobe_removed_intervals_merged:
        tobe_removed_word_idxs.extend(list(range(interval[0], interval[1])))
    
    words = [words[idx] for idx in range(len(words)) if idx not in tobe_removed_word_idxs]
    return " ".join(words).strip()


def main():
    text = "Một chuyến bay khởi hành từ Guadalajara, Mexico, tối 24/8, khi đến sân bay Guadalajara, các hành khách phát hiện tia lửa bắn ra từ động cơ bên phải. Một số hành khách cho biết đã nghe thấy âm thanh của một vụ nổ. Chuyến bay dự kiến kéo dài 3 tiếng, tuy nhiên, khoảng 10 phút sau khi cất cánh, hành khách phát hiện tia lửa bắn ra từ động cơ bên phải của máy bay. Theo hãng hàng không này, các hành khách trên chuyến bay đã được đưa đến một khách sạn và tiếp tục hành trình vào sáng hôm sau. Rất may mắn, vụ việc này không biến thành một thảm họa hàng không. Vài phút sau khi chiếc máy bay Airbus A320 cất cánh lúc 22h ngày 23/8, một tiếng nổ lớn đã vang lên và lửa bắt đầu phụt ra từ động cơ bên phải. Vụ cháy khiến các hành khách hoảng loạn, khóc lóc, la hét và cầu nguyện. Không ai trong số các thành viên phi hành đoàn hoặc 186 hành khách bị thương."
    out_text = block_repeat_ngrams(text, n=4)


if __name__ == "__main__":
    main()
