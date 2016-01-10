$(function() {
    var $story        = $('#story'),
        $question     = $('#question'),
        $answer       = $('#answer'),
        $getAnswer    = $('#get_answer'),
        $getStory     = $('#get_story'),
        $explainTable = $('#explanation');

    getStory();

    // Activate tooltip
    $('.qa-container').find('.glyphicon-info-sign').tooltip();

    $getAnswer.on('click', function(e) {
        e.preventDefault();
        getAnswer();
    });

    $getStory.on('click', function(e) {
        e.preventDefault();
        getStory();
    });

    function getStory() {
        $.get('/get/story', function(json) {
            $story.val(json["story"]);
            $question.val(json["question"]);
            $question.data('question_idx', json["question_idx"]);
            $question.data('suggested_question', json["question"]); // Save suggested question
            $answer.val('');
            $answer.data('correct_answer', json["correct_answer"]);
            //$explainTable.find('tbody').empty();
        });
    }

    function getAnswer() {
        var questionIdx       = $question.data('question_idx'),
            correctAnswer     = $answer.data('correct_answer'),
            suggestedQuestion = $question.data('suggested_question'),
            question          = $question.val();

        var userQuestion = suggestedQuestion !== question? question : '';
        var url = '/get/answer?question_idx=' + questionIdx +
            '&user_question=' + encodeURIComponent(userQuestion);

        $.get(url, function(json) {
            var predAnswer = json["pred_answer"],
                predProb = json["pred_prob"],
                memProbs = json["memory_probs"];

            var outputMessage = "Answer = '" + predAnswer + "'" +
                "\nConfidence score = " + (predProb * 100).toFixed(2) + "%";

            // Show answer's feedback only if suggested question was used
            if (userQuestion === '') {
                if (predAnswer === correctAnswer)
                    outputMessage += "\nCorrect!";
                else
                    outputMessage += "\nWrong. The correct answer is '" + correctAnswer + "'";
            }
            $answer.val(outputMessage);

            // Explain answer
            var explanationHtml = [];
            var sentenceList = $story.val().split('\n');
            var maxLatestSents = memProbs.length;
            var numSents = sentenceList.length;

            for (var i = Math.max(0, numSents - maxLatestSents); i < numSents; i++) {
                var rowHtml = [];
                rowHtml.push('<tr>');
                rowHtml.push('<td>' + sentenceList[i] + '</td>');
                for (var j = 0; j < 3; j++) {
                    var val = memProbs[i][j].toFixed(2);
                    if (val > 0) {
                        rowHtml.push('<td style="color: black; ' +
                            'background-color: rgba(97, 152, 246, ' + val + ');">' + val + '</td>');
                    } else {
                        rowHtml.push('<td style="color: black;">' + val + '</td>');
                    }
                }
                rowHtml.push('</tr>');
                explanationHtml.push(rowHtml.join('\n'));
            }
            $explainTable.find('tbody').html(explanationHtml);
        });
    }
});
