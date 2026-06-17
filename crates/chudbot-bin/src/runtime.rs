use std::net::SocketAddr;

use chudbot_bot::{BotRunOptions, BotRuntime, BotRuntimeTypes};
use chudbot_web::{WebRuntimeTypes, WebState};
use tokio::task::JoinError;
use tokio_util::sync::CancellationToken;

use crate::SHUTDOWN_GRACE_PERIOD;
use crate::errors::BinError;

/// Run fully constructed bot and web services until a process shutdown signal
/// or either service exits.
pub async fn run_runtime_services<R>(
    bot: BotRuntime<R>,
    web: WebState<R>,
    listen: SocketAddr,
) -> Result<(), BinError>
where
    R: BotRuntimeTypes + WebRuntimeTypes + 'static,
{
    let shutdown = CancellationToken::new();
    let bot_shutdown = shutdown.child_token();
    let web_shutdown = shutdown.child_token();
    let mut bot_task = tokio::spawn(async move {
        bot.run_with_options(
            bot_shutdown,
            BotRunOptions {
                drain_timeout: SHUTDOWN_GRACE_PERIOD,
            },
        )
        .await
        .map_err(BinError::Bot)
    });
    let mut web_task = tokio::spawn(async move {
        chudbot_web::run_until_shutdown(web, listen, async move {
            web_shutdown.cancelled().await;
        })
        .await
        .map_err(BinError::Web)
    });

    let mut bot_result = None;
    let mut web_result = None;
    tokio::select! {
        _ = shutdown_signal() => {
            tracing::info!("process shutdown requested");
            shutdown.cancel();
        }
        result = &mut bot_task => {
            tracing::info!("bot service exited; shutting down remaining services");
            shutdown.cancel();
            bot_result = Some(join_service_result("bot", result));
        }
        result = &mut web_task => {
            tracing::info!("web service exited; shutting down remaining services");
            shutdown.cancel();
            web_result = Some(join_service_result("web", result));
        }
    }

    let bot_result = match bot_result {
        Some(result) => result,
        None => join_service_result("bot", bot_task.await),
    };
    let web_result = match web_result {
        Some(result) => result,
        None => join_service_result("web", web_task.await),
    };
    bot_result?;
    web_result?;
    tracing::info!("all services stopped");
    Ok(())
}

fn join_service_result(
    task: &'static str,
    result: Result<Result<(), BinError>, JoinError>,
) -> Result<(), BinError> {
    match result {
        Ok(result) => result,
        Err(source) => {
            log_service_join_error(task, &source);
            Err(BinError::TaskJoin { task, source })
        }
    }
}

fn log_service_join_error(service: &'static str, error: &JoinError) {
    if error.is_cancelled() {
        tracing::warn!(service, error = %error, "service task was cancelled");
    } else if error.is_panic() {
        tracing::error!(service, error = %error, "service task panicked");
    } else {
        tracing::warn!(service, error = %error, "service task join failed");
    }
}

/// Wait for SIGINT or SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("received SIGINT"),
        () = terminate => tracing::info!("received SIGTERM"),
    }
}
