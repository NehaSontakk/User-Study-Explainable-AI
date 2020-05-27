from flask import Flask, render_template, session, request, flash, redirect, url_for
from config import Config
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, RadioField
from wtforms.validators import DataRequired, Required
from wtforms.validators import InputRequired, ValidationError
from werkzeug.datastructures import MultiDict
import pickle
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import pygal

from pygal.style import Style
custom_style = Style(
  background='transparent',
  plot_background='transparent',
  label_font_family='sans serif',
  )

app = Flask(__name__)
app.config.from_object(Config)
app.config["CACHE_TYPE"] = "null"
#feature list

feature = ["age","dependents","annual income","monthly loans","income stability","portfolio status","investment objective","duration","comfort","behaviour"]
#data for lime custom
file = open("user_risk_dtree.sav","rb")
dtree =  pickle.load(file)
df_riskcat = pd.read_csv("user_risk.csv")
df_riskcat.drop(columns=['output'],inplace=True)
X = df_riskcat.drop(columns=['categories'])
Y = df_riskcat['categories']
explainer = lime.lime_tabular.LimeTabularExplainer(X.to_numpy(), feature_names=feature, verbose=True, kernel_width=5)
predict_func_dtree = lambda x: dtree_model.predict_proba(x).astype(float)
#print(app.config)
mutual_fund = pd.read_excel("static/Mutual_Funds_dataset.xlsx")

class LoginForm(FlaskForm):
	username = StringField('Username\t',validators=[InputRequired()])
	age = RadioField('1. What is your age (in years)? ', choices=[('1','18-35'),('0.5','35-55'),('0.2','55+')], default=0, validators=[DataRequired()])
	dependents = RadioField('2. How many people are financially dependent on you?',choices=[('1','No One'),('0.8','Spouse Only'),('0.6','Parents Only'),('0.4','Spouse and Children'),('0.1','Parents, Spouse and Children.')],default=0,validators=[DataRequired()])
	annual_income = RadioField('3. What is your annual income?',choices=[('0.2','Below 1 Lakh'),('0.4','1 to 5 Lakh'),('0.6','5 to 10 Lakh'),('0.8','10 to 25 Lakh'),('1','Above 25 Lakh')],default=0,validators=[DataRequired()])
	monthly_loans = RadioField('4. What % of your monthly income goes into EMIs/outstanding loans?',choices=[('1','None'),('0.8','Upto 20%'),('0.6','20-30%'),('0.4','30-40%'),('0.2','Above 50%')],default=0,validators=[DataRequired()])
	income_stability = RadioField('5. What is the stability of your income?',choices=[('1','Guaranteed Stability'),('0.8','Highly Stable'),('0.6','Moderately Stable'),('0.4','Low Stability'),('0.1','Very Low Stability/Volatile')],default=0,validators=[DataRequired()])
	portfolio = RadioField('6. Where is most of your current portfolio parked?',choices=[('1','Stock Market'),('0.8','Mutual Funds'),('0.6','Bonds/Debt'),('0.4','Savings/Fixed deposits'),('0.1','Real Estate/Gold')],default=0,validators=[DataRequired()])
	investment_obj = RadioField('7. What is your primary investment objective?',choices=[('1','Wealth Creation'),('0.8','Monthly Income'),('0.6','Capital Preservation'),('0.4','Retirement Planning'),('0.1','Tax Saving')],default=0,validators=[DataRequired()])
	duration = RadioField('8. How long do you plan to stay invested?',choices=[('1','3-5 years'),('0.8','Less than 1 year'),('0.6','1-3 years'),('0.5','5-10 years'),('0.3','10+ years')],default=0,validators=[DataRequired()])
	comfort = RadioField('9. To achieve high returns, you are comfortable with high risk investments?',choices=[('1','Strongly Agree'),('0.8','Agree'),('0.5','Neutral'),('0.3','Disagree'),('0.1','Strongly Disagree')],default=0,validators=[DataRequired()])
	behaviour = RadioField('10. If you lose 20% of your investment in the first month, you will?',choices=[('1','Invest more'),('0.8','Keep investments as they are'),('0.6','Wait till market recovers and then sell'),('0.4','Sell and move cash to fixed deposits or liquid funds'),('0.2','Sell and preserve cash')],default=0,validators=[DataRequired()])

def riskcalc(form):
	polynomial_output = 1*float(form.age.data)+0.83*float(form.dependents.data)+0.83*float(form.age.data)*float(form.annual_income.data)+0.65*float(form.age.data)*float(form.dependents.data)*float(form.monthly_loans.data)+ 0.65*float(form.income_stability.data) + 0.5*float(form.portfolio.data)*float(form.age.data)*float(form.dependents.data)*float(form.duration.data)+0.8*float(form.investment_obj.data)*float(form.age.data)*float(form.dependents.data)+0.8*float(form.duration.data)*float(form.investment_obj.data)*float(form.age.data)*float(form.dependents.data)+0.7*float(form.comfort.data)*float(form.age.data)*float(form.dependents.data) + 0.65*float(form.behaviour.data)*float(form.age.data)*float(form.dependents.data)*float(form.income_stability.data)*float(form.duration.data)
	return polynomial_output

def riskcategory(risk_score):
	if risk_score<1.5:
		risk_cat = "No Risk"
	elif 1.6<= risk_score <= 2.3:
		risk_cat = "Low Risk"
	elif 2.4<= risk_score <= 3.3:
		risk_cat = "Moderate Risk"
	elif 3.4<= risk_score <= 4.3:
		risk_cat = "Likes Risk"
	elif 4.4<= risk_score:
		risk_cat= "High Risk"
	else:
		risk_cat = "error"
	return risk_cat


def lime_plot2(form):
	user_data = np.array([form.age.data,form.dependents.data,form.annual_income.data,form.monthly_loans.data,form.income_stability.data,form.portfolio.data,form.investment_obj.data,form.duration.data,form.comfort.data,form.behaviour.data])
	exp = explainer.explain_instance(np.float64(user_data), dtree.predict_proba,num_features=10)
	user_list =pd.DataFrame(exp.as_list(),columns=["Features/Variables","values"])
	chart = pygal.HorizontalBar(style=custom_style)
	chart.x_labels = user_list['Features/Variables'].to_list()
	chart.add('Features', user_list['values'].to_list())
	chart_data = chart.render_data_uri()
	return chart_data


def mutual_fund_rec(form,risk_cat):
	df_risk = mutual_fund.loc[mutual_fund['Risk_category'].astype(str) == risk_cat]	
	if df_risk.empty == True:
		return None
	else:
		df_dur = df_risk.loc[df_risk['Duration_conversion']==float(form.duration.data)]
		if df_dur.empty == True:
			df_risk_selected = df_risk.sample(3,replace=True)
			return df_risk_selected[['Name','Category','AUM', '6M_return', '1Y_return','Sharpe','sortino','Alpha','Beta','Min_inv','expense_ratio', 'P/B',
       'P/E', 'Peer_rank','Risk_category']]
		else:
			df_dur_selected = df_dur.sample(3,replace=True)
			return df_dur_selected[['Name','Category','AUM', '6M_return', '1Y_return','Sharpe','sortino','Alpha','Beta','Min_inv','expense_ratio', 'P/B',
       'P/E', 'Peer_rank','Risk_category']]
 
@app.route("/",methods=['GET', 'POST'])
def questionnaire():
	form = LoginForm()
	risk_score = False
	risk_cat = False
	df_return = pd.DataFrame()
	#chartjs
	lime_data = None
	if form.validate_on_submit():
		#print(form.validate_on_submit())
		risk_score = riskcalc(form)
		risk_cat = riskcategory(np.float64(risk_score))
		lime_data = lime_plot2(form)
		if form.is_submitted() == True:
			df_return = mutual_fund_rec(form,risk_cat)
		return render_template("questionnaire.html",form=form ,risk_score=risk_score ,risk_cat=risk_cat,data=df_return.to_html(index=False),lime_data=lime_data)

	else:
		risk_score = 'Please fill all fields in the questionnaire'
	
		return render_template("questionnaire.html",form=form ,risk_score=risk_score ,risk_cat=risk_cat,data=df_return.to_html(),lime_data=lime_data)




@app.route("/risktrends")
def risktrends():
	return render_template("risktrends.html")



@app.route("/mutualfundtrends")
def mutualfundtrends():
	return render_template("mutualfundtrends.html")

if __name__=="__main__":
	app.run(debug=True)
	
