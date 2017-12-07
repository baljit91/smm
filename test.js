/**
 * Created by Singh on 12/3/17.
 */
       $(document).ready(function(){



         easy_questions = [];
         medium_questions = [];
         hard_questions = [];
         var score = 0;

         

          


        $.ajax({
              url      : 'test.php',
              method   : 'post',
              data     : {id: 6},
              datatype : "JSON",
              success  : function(data){
                            // now update user record in table
                    console.log(data);
                    var itemData = $.parseJSON(data);

                    $.each(itemData, function(index) {
                        var que_id = parseInt(itemData[index].question_id);
                        var que_level = itemData[index].level;


                        if(que_level === "easy"){
                          easy_questions.push(que_id);
                        }
                        else if(que_level === "medium"){
                            medium_questions.push(que_id);
                        }
                        else{
                            hard_questions.push(que_id);
                        }

                     });


                    for (var i = easy_questions.length - 1; i > 0; i--) {
                        var j = Math.floor(Math.random() * (i + 1));
                        var temp = easy_questions[i];
                        easy_questions[i] = easy_questions[j];
                        easy_questions[j] = temp;
                    }

                    for (var i = medium_questions.length - 1; i > 0; i--) {
                        var j = Math.floor(Math.random() * (i + 1));
                        var temp = medium_questions[i];
                        medium_questions[i] = medium_questions[j];
                        medium_questions[j] = temp;
                    }


                    for (var i = hard_questions.length - 1; i > 0; i--) {
                        var j = Math.floor(Math.random() * (i + 1));
                        var temp = hard_questions[i];
                        hard_questions[i] = hard_questions[j];
                        hard_questions[j] = temp;
                    }



                    console.log(easy_questions.length );
                    console.log(medium_questions.length );
                    console.log(hard_questions.length );


              },
              async: false // <- this turns it into synchronous
          });


        

        //shuffle arrays

        

        



        total = 5;
        total_easy = 1;
        total_easy_correct = 0;
        total_medium = 0;
        total_medium_correct = 0;
        total_hard = 0;
        total_hard_correct = 0;

        start = 1;
        correct = 0;
        wrong = 0;
        unanswered = 0;
        current_que_id = easy_questions[0];

        question = '';
        option_1 = '';
        option_2 = '';
        option_3 = '';
        option_4 = '';
        current_level = '';
        correct_answer = '';



        //append ques
        $.ajax({
              url      : 'get_question.php',
              method   : 'post',
              data     : {que_id: current_que_id},
              datatype : "JSON",
              success  : function(data){
                            // now update user record in table
                    var itemData = $.parseJSON(data);
                        question = itemData.question;
                        option_1 = itemData.option_1;
                        option_2 = itemData.option_2;
                        option_3 = itemData.option_3;
                        option_4 = itemData.option_4;
                        correct_answer = itemData.answer;
                        current_level = itemData.level;


              },
              async: false // <- this turns it into synchronous
          });

        $('#que').text(question);
        $('#op1').text(option_1);
        $('#op2').text(option_2);
        $('#op3').text(option_3);
        $('#op4').text(option_4);



 //  append values in input fields
      $(document).on('click','a[data-role=next]',function(){
            start = start + 1;
            //if start == total display results and store
            

            user_answer = $('input[name=options]:checked').val();


            if(user_answer=== undefined){
              unanswered = unanswered + 1;
            }

            if(current_level === "easy" && user_answer === correct_answer ){
              
              easy_questions.splice( $.inArray(current_que_id,easy_questions) ,1 );
              current_que_id = medium_questions[0];
              correct = correct + 1;
              total_easy_correct = total_easy_correct + 1;
              score = score + 2;
            }
            else if(current_level === "easy" &&  user_answer !== correct_answer ) {
              easy_questions.splice( $.inArray(current_que_id,easy_questions) ,1 );
              current_que_id = easy_questions[0];
              if(user_answer !== undefined){
                wrong = wrong + 1;
              }
            }
            else if(current_level === "medium" &&  user_answer === correct_answer ) {
              medium_questions.splice( $.inArray(current_que_id,medium_questions) ,1 );
              current_que_id = hard_questions[0];
              correct = correct + 1;
              total_medium_correct = total_medium_correct + 1;
              score = score + 3;
            }
            else if(current_level === "medium" &&  user_answer !== correct_answer ) {
              medium_questions.splice( $.inArray(current_que_id,medium_questions) ,1 );
              current_que_id = easy_questions[0];
              if(user_answer !== undefined){
                wrong = wrong + 1;
              }
            }
            else if(current_level === "hard" &&  user_answer === correct_answer ) {
              hard_questions.splice( $.inArray(current_que_id,hard_questions) ,1 );
              current_que_id = hard_questions[0];
              correct = correct + 1;
              total_hard_correct = total_hard_correct + 1;
              score = score + 5;
            }
            else if(current_level === "hard" &&  user_answer !== correct_answer ) {
              hard_questions.splice( $.inArray(current_que_id,hard_questions) ,1 );
              current_que_id = medium_questions[0];
              if(user_answer !== undefined){
                wrong = wrong + 1;
              }
            }




            if(start > total){
              
              display_save();
              return;
            }



            if(current_level === "easy"){
                total_easy = total_easy + 1;
            }
            else if(current_level === "medium"){
                total_medium = total_medium + 1;
            }
            else{
                total_hard = total_hard + 1;
            }

             //append ques
        $.ajax({
              url      : 'get_question.php',
              method   : 'post',
              data     : {que_id: current_que_id},
              datatype : "JSON",
              success  : function(data){
                            // now update user record in table
                    var itemData = $.parseJSON(data);
                        question = itemData.question;
                        option_1 = itemData.option_1;
                        option_2 = itemData.option_2;
                        option_3 = itemData.option_3;
                        option_4 = itemData.option_4;
                        correct_answer = itemData.answer;
                        current_level = itemData.level;


              },
              async: false // <- this turns it into synchronous
          });

        $('#que').text(question);
        $('#op1').text(option_1);
        $('#op2').text(option_2);
        $('#op3').text(option_3);
        $('#op4').text(option_4);
        $('input[name=options]:checked').prop('checked', false);


            //get answer ajax

            //get que ajax
            
      });
      

      function display_save() {
        if(start <= total){
          unanswered = total - wrong - correct;
        }


        //display results
              $('input[name=options]').css("display", "none");
              $('#que').text("Test Result is Below !!");
              $('#op1').text("Total Questions : " + total);
              $('#op2').text("Correct Answers : " + correct);
              $('#op3').text("Wrong Answers : " + wrong);
              $('#op4').text("Unanswered Questions : " + unanswered);
              $('#score').css('display','block');
               $('#score').text("Your Score : " + score);



               //save result


               $.ajax({
              url      : 'save_results.php',
              method   : 'post',
              data     : { score : score , total_questions: total ,
                           correct: correct , wrong: wrong , easy_correct: total_easy_correct ,
                            medium_correct: total_medium_correct ,
                           hard_correct: total_hard_correct , unanswered: unanswered , total_easy: total_easy ,
                            total_medium: total_medium ,
                           total_hard: total_hard },
              success  : function(response){
                            // now update user record in table
                            

                         }
          });


              $('#chartContainer').css("display", "block");
              var chart = new CanvasJS.Chart("chartContainer", {
                animationEnabled: true,
                title: {
                  text: "Your Performance analysis"
                },
                data: [{
                  type: "pie",
                  startAngle: 240,
                  yValueFormatString: "##0.00\"%\"",
                  indexLabel: "{label} {y}",
                  dataPoints: [
                    {y: (correct/total)*100, label: "Correct"},
                    {y: (wrong/total)*100, label: "Wrong"},
                    {y: (unanswered/total)*100, label: "Unanswered"},
                    
                  ]
                }]
              });
              chart.render();


              //second chart
              $('#chartContainer2').css("display", "block");
              var chart = new CanvasJS.Chart("chartContainer2", {
                  animationEnabled: true,
                  title:{
                    text: "Performance Analysis"
                  },  
                  axisY: {
                    title: "Number of Questions",
                    titleFontColor: "#4F81BC",
                    lineColor: "#4F81BC",
                    labelFontColor: "#4F81BC",
                    tickColor: "#4F81BC"
                  },
                   
                  
                  data: [{
                    type: "column",
                    name: "Correct",
                    legendText: "Correct",
                    showInLegend: true, 
                    dataPoints:[
                      { label: "Easy", y: 266.21 },
                      { label: "Medium", y: 302.25 },
                      { label: "Hard", y: 157.20 }
                      
                    ]
                  },
                  {
                    type: "column", 
                    name: "Wrong",
                    legendText: "Wrong",
                    axisYType: "secondary",
                    showInLegend: true,
                    dataPoints:[
                      { label: "Easy", y: 10.46 },
                      { label: "Medium", y: 2.27 },
                      { label: "Hard", y: 3.99 }
                     
                    ]
                  }]
                });
                chart.render();


    }
      // now create event to get data from fields and update in database
//setTimeout('checkTime()',1000);



function c(){
  //give minutes here
  var n= 2;
  n = 60*n;
    var c=n;
    var min = parseInt(c/60);
    var sec = parseInt(c%60);

  $('.c').text(c);
  setInterval(function(){
    c--;
    min = parseInt(c/60);
      sec = parseInt(c%60);
    if(c>=0){
      $('.c').text(c);
      $('.min').text(min);
      $('.sec').text(sec);
    }
        if(c==0){
            display_save();
            return;
            $('.c').text(n);
        }

  },1000);
}


// Start
c();

       
});
