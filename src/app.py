import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.graph_objs as go
import numpy as np
from scipy import stats

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Dictionary containing distribution formulas
distribution_formulas = {
    'Uniform': r'f(x) = \frac{1}{b-a} \text{ for } a \leq x \leq b',
    'Exponential': r'f(x) = \lambda e^{-\lambda x} \text{ for } x \geq 0',
    'Normal': r'f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}',
    'Poisson': r'P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}',
    'Geometric': r'P(X = k) = p(1-p)^{k-1}',
    'Binomial': r'P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}',
    'Negative Binomial': r'P(X = k) = \binom{k+r-1}{k}p^r(1-p)^k',
    'Gamma': r'f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}',
    'Beta': r'f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}',
    'Chi-square': r'f(x) = \frac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}e^{-x/2}',
    'Pareto': r'f(x) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}',
    'Student\'s T': r'f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})}(1+\frac{x^2}{\nu})^{-\frac{\nu+1}{2}}',
    'Weibull': r'f(x) = \frac{k}{\lambda}(\frac{x}{\lambda})^{k-1}e^{-(x/\lambda)^k}'
}

# Dictionary mapping distributions to their parameter inputs
distribution_params = {
    'Uniform': [
        {'name': 'a (minimum)', 'id': 'a', 'default': 0},
        {'name': 'b (maximum)', 'id': 'b', 'default': 1}
    ],
    'Exponential': [
        {'name': 'λ (rate)', 'id': 'lambda', 'default': 1}
    ],
    'Normal': [
        {'name': 'μ (mean)', 'id': 'mu', 'default': 0},
        {'name': 'σ (std dev)', 'id': 'sigma', 'default': 1}
    ],
    'Poisson': [
        {'name': 'λ (rate)', 'id': 'lambda', 'default': 1}
    ],
    'Geometric': [
        {'name': 'p (probability)', 'id': 'p', 'default': 0.5}
    ],
    'Binomial': [
        {'name': 'n (trials)', 'id': 'n', 'default': 10},
        {'name': 'p (probability)', 'id': 'p', 'default': 0.5}
    ],
    'Negative Binomial': [
        {'name': 'r (successes)', 'id': 'r', 'default': 5},
        {'name': 'p (probability)', 'id': 'p', 'default': 0.5}
    ],
    'Gamma': [
        {'name': 'α (shape)', 'id': 'alpha', 'default': 2},
        {'name': 'β (rate)', 'id': 'beta', 'default': 1}
    ],
    'Beta': [
        {'name': 'α (shape)', 'id': 'alpha', 'default': 2},
        {'name': 'β (shape)', 'id': 'beta', 'default': 2}
    ],
    'Chi-square': [
        {'name': 'k (degrees of freedom)', 'id': 'df', 'default': 3}
    ],
    'Pareto': [
        {'name': 'α (shape)', 'id': 'alpha', 'default': 3},
        {'name': 'xₘ (scale)', 'id': 'xm', 'default': 1}
    ],
    'Student\'s T': [
        {'name': 'ν (degrees of freedom)', 'id': 'df', 'default': 3}
    ],
    'Weibull': [
        {'name': 'k (shape)', 'id': 'k', 'default': 1.5},
        {'name': 'λ (scale)', 'id': 'lambda', 'default': 1}
    ]
}

# App layout
app.layout = html.Div([
    html.H1('Probability Distribution Visualizer'),
    
    # Distribution selection dropdown
    html.Div([
        html.Label('Select Distribution:'),
        dcc.Dropdown(
            id='distribution-dropdown',
            options=[{'label': dist, 'value': dist} for dist in distribution_formulas.keys()],
            value='Normal'
        )
    ]),
    
    # Formula display
    html.Div([
        html.Label('Distribution Formula:'),
        dcc.Markdown(id='formula-display', mathjax=True)
    ]),
    
    # Parameters input section
    html.Div(id='params-container'),
    
    # Generate button
    html.Button('Generate Plots', id='generate-button', n_clicks=0),
    
    # Plots
    html.Div([
        dcc.Graph(id='pdf-pmf-plot'),
        dcc.Graph(id='cdf-cmf-plot')
    ]),
    
    # Sampling Distribution Section
    html.Div([
        html.H2('Sampling Distribution'),
        html.P('Demonstrates the Central Limit Theorem by showing the distribution of sample means'),
        html.Div([
            html.Label('Number of samples (N):'),
            dcc.Input(
                id='sample-size-input',
                type='number',
                value=1000,
                min=100,
                max=10000,
                step=100
            ),
            html.Label('Sample group size:'),
            dcc.Input(
                id='group-size-input',
                type='number',
                value=30,
                min=1,
                max=100,
                step=1
            ),
            html.Button('Generate Sampling Distribution', id='generate-sampling-button', n_clicks=0)
        ], style={'marginBottom': '20px'}),
        dcc.Graph(id='sampling-dist-plot')
    ])
])

@app.callback(
    Output('formula-display', 'children'),
    Input('distribution-dropdown', 'value')
)
def update_formula(distribution):
    return f'${distribution_formulas[distribution]}$'

@app.callback(
    Output('params-container', 'children'),
    Input('distribution-dropdown', 'value')
)
def update_params(distribution):
    params = distribution_params[distribution]
    return [
        html.Div([
            html.Label(param['name']),
            dcc.Input(
                id={'type': 'dynamic-param', 'index': param['id']},
                type='number',
                value=param['default'],
                step=0.1
            )
        ]) for param in params
    ]

def get_x_range(dist_name, params):
    if dist_name == 'Uniform':
        a, b = params
        margin = (b - a) * 0.1  # Add 10% margin on each side
        return np.linspace(a - margin, b + margin, 200)
    elif dist_name == 'Exponential':
        rate = params[0]
        mean = 1/rate
        return np.linspace(0, mean * 5, 200)  # Show up to 5 times the mean
    elif dist_name == 'Normal':
        mu, sigma = params
        return np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    elif dist_name == 'Poisson':
        lambda_param = params[0]
        return np.arange(0, max(20, int(lambda_param * 2 + 4 * np.sqrt(lambda_param))))
    elif dist_name == 'Geometric':
        p = params[0]
        mean = 1/p
        return np.arange(1, int(mean + 4 * np.sqrt((1-p)/p**2)))
    elif dist_name == 'Binomial':
        n, p = params
        mean = n * p
        std = np.sqrt(n * p * (1-p))
        return np.arange(max(0, int(mean - 3*std)), min(n, int(mean + 3*std)) + 1)
    elif dist_name == 'Negative Binomial':
        r, p = params
        mean = r * (1-p)/p
        std = np.sqrt(r * (1-p)/p**2)
        return np.arange(0, int(mean + 4*std))
    elif dist_name == 'Gamma':
        alpha, beta = params
        mean = alpha/beta
        std = np.sqrt(alpha/beta**2)
        return np.linspace(0, mean + 4*std, 200)
    elif dist_name == 'Beta':
        return np.linspace(0, 1, 200)
    elif dist_name == 'Chi-square':
        df = params[0]
        return np.linspace(0, max(20, df + 4*np.sqrt(2*df)), 200)
    elif dist_name == 'Pareto':
        alpha, xm = params
        if alpha > 1:
            mean = (alpha * xm)/(alpha - 1)
            if alpha > 2:
                std = xm * alpha / ((alpha-1) * np.sqrt((alpha-2)))
                return np.linspace(xm, mean + 4*std, 200)
        return np.linspace(xm, xm * 10, 200)
    elif dist_name == 'Student\'s T':
        df = params[0]
        if df > 2:
            std = np.sqrt(df/(df-2))
            return np.linspace(-4*std, 4*std, 200)
        return np.linspace(-10, 10, 200)
    elif dist_name == 'Weibull':
        k, lambda_param = params
        mean = lambda_param * np.math.gamma(1 + 1/k)
        var = lambda_param**2 * (np.math.gamma(1 + 2/k) - (np.math.gamma(1 + 1/k))**2)
        std = np.sqrt(var)
        return np.linspace(0, mean + 4*std, 200)

@app.callback(
    [Output('pdf-pmf-plot', 'figure'),
     Output('cdf-cmf-plot', 'figure'),
     Output('sampling-dist-plot', 'figure')],
    Input('generate-button', 'n_clicks'),
    [State('distribution-dropdown', 'value'),
     State({'type': 'dynamic-param', 'index': ALL}, 'value'),
     State('sample-size-input', 'value'),
     State('group-size-input', 'value')]
)
def update_plots(n_clicks, distribution, param_values, sample_size, group_size):
    if n_clicks == 0:
        return {}, {}, {}
    
    # Get the relevant parameters for the selected distribution
    dist_params = distribution_params[distribution]
    param_values = param_values[:len(dist_params)]
    
    # Generate x values based on distribution and parameters
    x = get_x_range(distribution, param_values)
    discrete = distribution in ['Poisson', 'Geometric', 'Binomial', 'Negative Binomial']
    
    # Calculate distribution values
    try:
        if distribution == 'Uniform':
            a, b = param_values
            if b <= a:
                raise ValueError("Maximum value (b) must be greater than minimum value (a)")
            dist = stats.uniform(loc=a, scale=b-a)
        elif distribution == 'Exponential':
            rate = param_values[0]
            if rate <= 0:
                raise ValueError("Rate parameter must be positive")
            dist = stats.expon(scale=1/rate)
        elif distribution == 'Normal':
            mu, sigma = param_values
            dist = stats.norm(mu, sigma)
        elif distribution == 'Poisson':
            lambda_param = param_values[0]
            dist = stats.poisson(lambda_param)
        elif distribution == 'Geometric':
            p = param_values[0]
            dist = stats.geom(p)
        elif distribution == 'Binomial':
            n, p = param_values
            dist = stats.binom(int(n), p)
        elif distribution == 'Negative Binomial':
            r, p = param_values
            dist = stats.nbinom(int(r), p)
        elif distribution == 'Gamma':
            alpha, beta = param_values
            dist = stats.gamma(alpha, scale=1/beta)
        elif distribution == 'Beta':
            alpha, beta = param_values
            dist = stats.beta(alpha, beta)
        elif distribution == 'Chi-square':
            df = param_values[0]
            dist = stats.chi2(df)
        elif distribution == 'Pareto':
            alpha, xm = param_values
            dist = stats.pareto(alpha, scale=xm)
        elif distribution == 'Student\'s T':
            df = param_values[0]
            dist = stats.t(df)
        elif distribution == 'Weibull':
            k, lambda_param = param_values
            dist = stats.weibull_min(k, scale=lambda_param)
            
        # Create PDF/PMF plot
        if discrete:
            pmf = dist.pmf(x)
            pdf_pmf_fig = {
                'data': [{'x': x, 'y': pmf, 'type': 'bar', 'name': 'PMF'}],
                'layout': {
                    'title': f'{distribution} Probability Mass Function',
                    'xaxis': {'title': 'k'},
                    'yaxis': {'title': 'P(X = k)', 'range': [0, max(pmf) * 1.05]}
                }
            }
        else:
            pdf = dist.pdf(x)
            pdf_pmf_fig = {
                'data': [{'x': x, 'y': pdf, 'type': 'line', 'name': 'PDF'}],
                'layout': {
                    'title': f'{distribution} Probability Density Function',
                    'xaxis': {'title': 'x'},
                    'yaxis': {'title': 'f(x)', 'range': [0, max(pdf) * 1.05]}
                }
            }
        
        # Create CDF plot
        cdf = dist.cdf(x)
        cdf_fig = {
            'data': [{'x': x, 'y': cdf, 'type': 'line', 'name': 'CDF'}],
            'layout': {
                'title': f'{distribution} Cumulative Distribution Function',
                'xaxis': {'title': 'x'},
                'yaxis': {'title': 'F(x)', 'range': [0, 1.05]}
            }
        }
        
        # Generate samples and calculate means for CLT demonstration
        random_samples = dist.rvs(size=(sample_size, group_size))
        sample_means = np.mean(random_samples, axis=1)
        
        # Calculate mean and std of sample means for reference normal distribution
        means_mean = np.mean(sample_means)
        means_std = np.std(sample_means)
        
        # Create x range for normal reference line
        x_means = np.linspace(means_mean - 4*means_std, means_mean + 4*means_std, 100)
        normal_ref = stats.norm.pdf(x_means, means_mean, means_std)
        
        # Create histogram of sample means
        sampling_fig = {
            'data': [
                {
                    'type': 'histogram',
                    'x': sample_means,
                    'name': 'Sample Means',
                    'histnorm': 'probability density',
                    'opacity': 0.7,
                    'nbinsx': 30
                },
                {
                    'type': 'scatter',
                    'x': x_means,
                    'y': normal_ref,
                    'name': 'Normal Reference',
                    'line': {'color': 'red', 'dash': 'dash'}
                }
            ],
            'layout': {
                'title': 'Sampling Distribution of Means',
                'xaxis': {'title': 'Sample Mean'},
                'yaxis': {'title': 'Density'},
                'showlegend': True,
                'bargap': 0.1,
            }
        }
        
        return pdf_pmf_fig, cdf_fig, sampling_fig
        
    except Exception as e:
        error_fig = {
            'data': [],
            'layout': {'title': f'Error: Invalid parameters - {str(e)}'}
        }
        return error_fig, error_fig, error_fig

if __name__ == '__main__':
    app.run_server(debug=True)